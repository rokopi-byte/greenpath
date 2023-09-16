import numpy as np
import torch
from copy import copy
from torch.distributions import Categorical
import warnings
import os
import sys
import re
import cv2
from shutil import copyfile
from . import rect_utils

warnings.filterwarnings("ignore", category=UserWarning)


def cutFixOnTarget(trajs, target_annos):
    task_names = np.unique([traj['task'] for traj in trajs])
    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            traj_len = get_num_step2target(traj['X'], traj['Y'], bbox)
            num_steps_task[i] = traj_len
            traj['X'] = traj['X'][:traj_len]
            traj['Y'] = traj['Y'][:traj_len]


def pos_to_action(center_x, center_y, patch_size, patch_num):
    x = center_x // patch_size[0]
    y = center_y // patch_size[1]

    if x >= patch_num[0]:
        x = patch_num[0] - 1

    if y >= patch_num[1]:
        y = patch_num[1] - 1

    return int(patch_num[0] * y + x)


def action_to_pos(acts, patch_size, patch_num):
    patch_y = acts // patch_num[0]
    patch_x = acts % patch_num[0]

    pixel_x = patch_x * patch_size[0] + patch_size[0] / 2
    pixel_y = patch_y * patch_size[1] + patch_size[1] / 2
    return pixel_x, pixel_y


def select_action(obs, policy, sample_action, action_mask=None, fog_of_war=None,
                  softmask=False, eps=1e-20, prev_pos=None, walkable=None):
    probs, values = policy(*obs)
    action_mask = None  # act_mask.copy()
    fog_of_war = None
    prev_pos = None
    walls = None
    #walls = walkable.clone()
    #walls[walkable == 0] = 1
    #walls[walkable != 0] = 0
    #walls = walls.view(walls.size(0), -1).to(torch.bool)

    if prev_pos is not None and action_mask is not None:
        action_mask = action_mask + walls + prev_pos
    elif action_mask is None and prev_pos is not None:
        action_mask = prev_pos
        # npact = action_mask.detach().cpu().numpy()
    if sample_action:
        m = Categorical(probs)
        if action_mask is not None:
            # prevent sample previous actions by re-normalizing probs
            probs_new = probs.clone().detach()
            if fog_of_war is not None:
                action_mask = action_mask + fog_of_war
                for i in range(len(action_mask)):
                    if (action_mask[i] == 0).sum().item() == 0:
                        action_mask[i] = fog_of_war[i]

            if softmask:
                probs_new = probs_new * action_mask
            else:
                probs_new[action_mask] = eps

            probs_new /= probs_new.sum(dim=1).view(probs_new.size(0), 1)

            m_new = Categorical(probs_new)
            actions = m_new.sample()
        else:
            actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs
    else:
        m = Categorical(probs)
        # probs_new = probs.clone().detach()
        # probs_new[action_mask.view(probs_new.size(0), -1)] = 0
        actions = torch.argmax(probs, dim=1)
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs


def bresenham(start, end):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    dx = x2 - x1
    dy = y2 - y1
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def get_walls_metric_intersection(actions, walls, patch_num, penalty=0.01):
    total_penalty = 0
    for i in range(len(actions)):
        actions_penalty = 0
        act = actions[i][0]
        x, y = (act // patch_num[0]).item(), (act % patch_num[0]).item()
        if walls[x][y] == 1:
            actions_penalty = actions_penalty + penalty
        if i > 0:
            x1, y1 = (actions[i - 1][0] // patch_num[0]).item(), (actions[i - 1][0] % patch_num[0]).item()
            brese = bresenham((x1, y1), (x, y))
            for p in brese:
                if p[0] >= patch_num[0] or p[1] >= patch_num[1]:
                    actions_penalty = actions_penalty + penalty
                elif walls[p[0]][p[1]] == 1:
                    actions_penalty = actions_penalty + penalty
            if len(brese) >= 6:
                actions_penalty = actions_penalty + (penalty * len(brese))
        total_penalty = total_penalty + actions_penalty
    return total_penalty, total_penalty / len(actions)


def create_action_matrix(x, y, h, w):
    xc = x.cpu().numpy()
    yc = y.cpu().numpy()
    Y, X = np.ogrid[:h, :w]
    dist = np.zeros(shape=(x.size(0), h, w))
    for i in range(x.size(0)):
        if xc[i] < 0 or yc[i] < 0:
            continue
        e = np.sqrt((xc[i] - X) ** 2 + (yc[i] - Y) ** 2)
        e = 1 - ((e - np.min(e)) / (np.max(e) - np.min(e)))
        e = e ** 4
        dist[i, :, :] = e
        if xc[i] >= h:
            xc[i] = w - 1
        if yc[i] >= w:
            yc[i] = w - 1
        dist[i, int(xc[i]), int(yc[i])] = 1
    return torch.from_numpy(dist).view(x.size(0), -1).float().cuda()


def create_action_matrix_real(act, h, w):
    a = act.cpu().numpy()
    xc, yc = (a // w), (a % w)
    Y, X = np.ogrid[:h, :w]
    dist = np.zeros(shape=(act.size(0), h, w))
    weights = np.zeros(shape=(act.size(0), h, w))
    for i in range(act.size(0)):
        if xc[i] < 0 or yc[i] < 0:
            continue
        e = np.sqrt((xc[i] - X) ** 2 + (yc[i] - Y) ** 2)
        e = 1 - ((e - np.min(e)) / (np.max(e) - np.min(e)))
        e = e ** 4
        dist[i, :, :] = e
        weights[i, :, :] = e + 0.1
        weights[i] = np.clip(weights[i], 0.1, 1.0)
        if xc[i] >= h:
            xc[i] = w - 1
        if yc[i] >= w:
            yc[i] = w - 1
        dist[i, int(xc[i]), int(yc[i])] = 1
        weights[i, int(xc[i]), int(yc[i])] = 1

    matrix = torch.from_numpy(dist).view(act.size(0), -1).float().cuda()
    weights = torch.from_numpy(weights).view(act.size(0), -1).float().cuda()
    return matrix, weights


def create_action_matrix_single(x, y, h, w):
    xc = x
    yc = y
    Y, X = np.ogrid[:h, :w]
    dist = np.zeros(shape=(h, w))
    e = np.sqrt((xc - X) ** 2 + (yc - Y) ** 2)
    e = 1 - ((e - np.min(e)) / (np.max(e) - np.min(e)))
    e = e ** 4
    dist[:, :] = e
    dist[int(xc), int(yc)] = 1
    return torch.from_numpy(dist).float().cuda()


def get_walls_loss(actions, walls, patch_num, penalty=0.01):
    total_penalty = 0
    for i in range(len(actions)):
        actions_penalty = 0
        act = actions[i]
        x, y = (act // patch_num[0]).item(), (act % patch_num[0]).item()
        if walls[i][0, x, y] == 1:
            actions_penalty = actions_penalty + penalty
        total_penalty = total_penalty + actions_penalty
    return total_penalty / len(actions)


def collect_trajs(env,
                  policy,
                  patch_num,
                  max_traj_length,
                  is_eval=False,
                  sample_action=True):
    rewards = []
    obs_fov = env.observe()
    act, log_prob, value, prob = select_action((obs_fov, env.task_ids),
                                               policy,
                                               sample_action,
                                               action_mask=env.action_mask.clone(),
                                               fog_of_war=env.fog_of_war.clone(),
                                               prev_pos=env.previous_position,
                                               walkable=env.states[:, 25])
    status = [env.status]
    values = [value]
    log_probs = [log_prob]
    fog_of_wars = [env.fog_of_war]
    SASPs = []
    penalties = []
    i = 0
    if is_eval:
        actions = []
        while i < max_traj_length:
            penalties.append(env.return_penalty_action(act))
            new_obs_fov, curr_status = env.step(act)
            status.append(curr_status)
            actions.append(act)

            obs_fov = new_obs_fov
            act, log_prob, value, prob_new = select_action(
                (obs_fov, env.task_ids),
                policy,
                sample_action,
                action_mask=env.action_mask,
                fog_of_war=env.fog_of_war,
                prev_pos=env.previous_position,
                walkable=env.states[:, 25])
            i = i + 1

        trajs = {
            'status': torch.stack(status),
            'actions': torch.stack(actions),
            'init': env.fixations[:, 0],
            'probs': prob_new
        }

    else:
        IORs = []
        IORs.append(
            env.action_mask.to(dtype=torch.float).view(env.batch_size, 1,
                                                       patch_num[1], -1))
        while i < max_traj_length and env.status.min() < 1:
            penalties.append(env.return_penalty_action(act))
            new_obs_fov, curr_status = env.step(act)
            fog_of_wars.append(env.fog_of_war)
            status.append(curr_status)
            SASPs.append((obs_fov, act, new_obs_fov))
            obs_fov = new_obs_fov.clone().detach()

            IORs.append(
                env.action_mask.to(dtype=torch.float).view(
                    env.batch_size, 1, patch_num[1], -1))

            act, log_prob, value, prob_new = select_action(
                (obs_fov, env.task_ids),
                policy,
                sample_action,
                action_mask=env.action_mask.clone(),
                fog_of_war=env.fog_of_war.clone(),
                walkable=env.states[:, 25])
            values.append(value)
            log_probs.append(log_prob)

            rewards.append(torch.zeros(env.batch_size))

            i = i + 1

        S = torch.stack([sasp[0] for sasp in SASPs])
        A = torch.stack([sasp[1] for sasp in SASPs])
        V = torch.stack(values)
        R = torch.stack(rewards)
        fogs = torch.stack(fog_of_wars)
        pen = torch.stack(penalties)
        LogP = torch.stack(log_probs[:-1])
        status = torch.stack(status[1:])

        bs = len(env.img_names)
        trajs = []

        for i in range(bs):
            ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
            if status[:, i].sum() == 0:
                ind = status.size(0)
            trajs.append({
                'curr_states': S[:ind, i],
                'actions': A[:ind, i],
                'values': V[:ind + 1, i],
                'log_probs': LogP[:ind, i],
                'rewards': R[:ind, i],
                'fogs': fogs[:ind, i],
                'penalties': pen[:ind, i],
                'task_id': env.task_ids[i].repeat(ind),
                'img_name': [env.img_names[i]] * ind,
                'length': ind
            })

    return trajs


def compute_return_advantage(rewards, values, gamma, mtd='CRITIC', tau=0.96):
    device = rewards.device
    acc_reward = torch.zeros_like(rewards, dtype=torch.float, device=device)
    acc_reward[-1] = rewards[-1]
    for i in reversed(range(acc_reward.size(0) - 1)):
        acc_reward[i] = rewards[i] + gamma * acc_reward[i + 1]

    # compute advantages
    if mtd == 'MC':  # Monte-Carlo estimation
        advs = acc_reward - values[:-1]
    elif mtd == 'CRITIC':  # critic estimation
        advs = rewards + gamma * values[1:] - values[:-1]
    elif mtd == 'GAE':  # generalized advantage estimation
        delta = rewards + gamma * values[1:] - values[:-1]
        adv = torch.zeros_like(delta, dtype=torch.float, device=device)
        adv[-1] = delta[-1]
        for i in reversed(range(delta.size(0) - 1)):
            adv[i] = delta[i] + gamma * tau * adv[i + 1]
    else:
        raise NotImplementedError

    return acc_reward.squeeze(), advs.squeeze()


def process_trajs(trajs, gamma, mtd='CRITIC', tau=0.96, max_length=30):
    # compute discounted cummulative reward
    device = trajs[0]['log_probs'].device
    avg_return = 0
    avg_penalties = 0
    avg_abs = 0
    for traj in trajs:

        acc_reward = torch.zeros_like(traj['rewards'],
                                      dtype=torch.float,
                                      device=device)
        acc_reward[-1] = traj['rewards'][-1]

        if traj['abs_rewards'] is not None and traj['abs_rewards'].size(0) > 0:
            acc_reward_abs = torch.zeros_like(traj['abs_rewards'],
                                              dtype=torch.float,
                                              device=device)

            acc_reward_abs[-1] = traj['abs_rewards'][-1]
            for i in reversed(range(acc_reward_abs.size(0) - 1)):
                acc_reward_abs[i] = traj['abs_rewards'][i] + (gamma ** (i+2)) * acc_reward_abs[i + 1]

            acc_reward[-1] = acc_reward[-1] + gamma * acc_reward_abs[0]

        for i in reversed(range(acc_reward.size(0) - 1)):
            acc_reward[i] = traj['rewards'][i] + gamma * acc_reward[i + 1]
            avg_penalties += traj['penalties'][i]

        traj['acc_rewards'] = acc_reward
        avg_return += gamma * acc_reward[0]

        if  traj['abs_rewards'] is not None and traj['abs_rewards'].size(0) > 0:
            avg_abs += acc_reward_abs[0]
            traj['rewards'][-1] += acc_reward_abs[0]

        values = traj['values']
        # compute advantages
        if mtd == 'MC':  # Monte-Carlo estimation
            traj['advantages'] = traj['acc_rewards'] - values[:-1]

        elif mtd == 'CRITIC':  # critic estimation
            traj['advantages'] = traj[
                                     'rewards'] + gamma * values[1:] - values[:-1]

        elif mtd == 'GAE':  # generalized advantage estimation
            delta = traj['rewards'] + gamma * values[1:] - values[:-1]
            adv = torch.zeros_like(delta, dtype=torch.float, device=device)
            adv[-1] = delta[-1]
            for i in reversed(range(delta.size(0) - 1)):
                adv[i] = delta[i] + gamma * tau * adv[i + 1]
            traj['advantages'] = adv
        else:
            raise NotImplementedError

    return avg_return / len(trajs), avg_penalties / len(trajs)#, avg_abs / len(trajs)


def get_num_step2target(X, Y, bbox, key=None):
    if len(X) < 29:
        return len(X)
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X >= bbox[0][0], X <= bbox[0][0] + bbox[0][2])
    on_target_Y = np.logical_and(Y >= bbox[0][1], Y <= bbox[0][1] + bbox[0][3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        # if key is not None:
        #     print("Key", key)
        # print("X", X, "Y", Y, "BBOX", bbox[0])
        return 1000  # some big enough number


def get_CDF(num_steps, max_step):
    cdf = np.zeros(max_step)
    total = float(len(num_steps))
    for i in range(1, max_step + 1):
        cdf[i - 1] = np.sum(num_steps <= i) / total
    return cdf


def get_num_steps(trajs, target_annos, task_names):
    num_steps = {}
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            step_num = get_num_step2target(traj['X'], traj['Y'], bbox, key)
            num_steps_task[i] = step_num
            traj['X'] = traj['X'][:step_num]
            traj['Y'] = traj['Y'][:step_num]
        num_steps[task] = num_steps_task
    return num_steps


def get_mean_cdf(num_steps, task_names, max_step):
    cdf_tasks = []
    for task in task_names:
        cdf_tasks.append(get_CDF(num_steps[task], max_step))
    return cdf_tasks


def compute_search_cdf(scanpaths, annos, max_step, return_by_task=False):
    # compute search CDF
    task_names = np.unique([traj['task'] for traj in scanpaths])
    num_steps = get_num_steps(scanpaths, annos, task_names)
    cdf_tasks = get_mean_cdf(num_steps, task_names, max_step + 1)
    if return_by_task:
        return dict(zip(task_names, cdf_tasks))
    else:
        mean_cdf = np.mean(cdf_tasks, axis=0)
        std_cdf = np.std(cdf_tasks, axis=0)
        return mean_cdf, std_cdf


def calc_overlap_ratio(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    patch_area = float(patch_size[0] * patch_size[1])
    aoi_ratio = np.zeros((1, patch_num[1], patch_num[0]), dtype=np.float32)
    b = bbox[0]
    tl_x, tl_y = b[0], b[1]
    br_x, br_y = b[0] + b[2], b[1] + b[3]
    lx, ux = tl_x // patch_size[0], br_x // patch_size[0]
    ly, uy = tl_y // patch_size[1], br_y // patch_size[1]

    for x in range(lx, ux + 1):
        for y in range(ly, uy + 1):
            patch_tlx, patch_tly = x * patch_size[0], y * patch_size[1]
            patch_brx, patch_bry = patch_tlx + patch_size[
                0], patch_tly + patch_size[1]

            aoi_tlx = tl_x if patch_tlx < tl_x else patch_tlx
            aoi_tly = tl_y if patch_tly < tl_y else patch_tly
            aoi_brx = br_x if patch_brx > br_x else patch_brx
            aoi_bry = br_y if patch_bry > br_y else patch_bry

            aoi_ratio[0, y, x] = max((aoi_brx - aoi_tlx), 0) * max(
                (aoi_bry - aoi_tly), 0) / float(patch_area)

    return aoi_ratio


def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return mask.astype(np.float32)


def multi_hot_coding(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    thresh = 0
    if len(bbox) == 0:
        print("Warning: bbox absent")
        bbox = [[0, 0, 1, 1]]
    aoi_ratio = calc_overlap_ratio(bbox, patch_size, patch_num)
    hot_ind = aoi_ratio > thresh
    while hot_ind.sum() == 0:
        thresh *= 0.8
        hot_ind = aoi_ratio > thresh

        if hot_ind.sum() == 0 and thresh == 0:
            break

    aoi_ratio[hot_ind] = 1
    aoi_ratio[np.logical_not(hot_ind)] = 0

    return aoi_ratio[0]


def actions2scanpaths(actions, patch_num, im_w, im_h, fixations=None):
    # convert actions to scanpaths
    scanpaths = []
    for traj in actions:
        task_name, img_name, condition, actions, init = traj
        actions = actions.to(dtype=torch.float32)
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        x0 = init[0].item() / float(patch_num[0])
        y0 = init[1].item() / float(patch_num[1])
        fixs = np.concatenate([np.array([[x0], [y0]]),
                               fixs.cpu().numpy()],
                              axis=1)
        # if fixations is not None:
        #    fixs = np.concatenate([np.array([[fixations[img_name[:-4]][0] / 46],
        #                                     [fixations[img_name[:-4]][1] / 46]]), fixs.cpu().numpy()], axis=1)
        # else:
        scanpaths.append({
            'X': fixs[0] * im_w[img_name],
            'Y': fixs[1] * im_h[img_name],
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths


def preprocess_fixations(trajs,
                         patch_size,
                         patch_num,
                         im_h,
                         im_w,
                         truncate_num=-1,
                         need_label=True):
    fix_labels = []
    abs_states = []
    first_fixations = {}
    for traj in trajs:
        fixs = []
        if truncate_num < 1:
            traj_len = len(traj['X'])
        else:
            traj_len = min(truncate_num, len(traj['X']))

        for i in range(0, traj_len):
            label = pos_to_action(traj['X'][i], traj['Y'][i], patch_size[traj['name']],
                                  patch_num)

            tar_x, tar_y = action_to_pos(label, patch_size[traj['name']], patch_num)

            if i != 0:
                fix_label = (traj['name'], traj['task'], copy(fixs), label, (tar_x, tar_y))
            else:
                first_fixations.setdefault(traj["name"] + "_" + traj['task'], []).append((traj['X'][i], traj['Y'][i]))
                first_fixations.setdefault(traj["name"], []).append((traj['X'][i], traj['Y'][i]))

            # discretize fixations

            fixs.append((tar_x, tar_y))

            if i != 0:
                fix_labels.append(fix_label)

        diff = truncate_num - traj_len
        if diff > 0:
            tar_x, tar_y = fixs[-1]
            for i in range(0, diff):
                label = pos_to_action(tar_x, tar_y, patch_size[traj['name']], patch_num)
                fix_label = (traj['name'], traj['task'], copy(fixs), label, (tar_x, tar_y))
                abs_states.append(fix_label)
    return fix_labels, first_fixations, abs_states


def _file_at_step(step, name):
    return "save_{}_{}k{}.pkg".format(name, int(step // 1000),
                                      int(step % 1000))


def show_states(states, name="states", action=None, chan=25):
    # num_of_states = states.size(0)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    img = np.zeros((states.size(1), states.size(1)), dtype=np.uint8)

    m = states[chan, :, :] > 0.6

    # for i in range(states.size(0)):
    m = m * 255

    # m[5, 5] = 0
    img = img + m.detach().cpu().numpy()  # cv2.resize(m.detach().cpu().numpy(), (states.size(1), states.size(1)*10), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.uint8)
    if action:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[action[1], action[0]] = (0, 0, 255)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def _file_best(name):
    return "trained_{}.pkg".format(name)


def save(global_step,
         model,
         optim,
         name,
         pkg_dir="",
         is_best=False,
         max_checkpoints=None):
    if optim is None:
        raise ValueError("cannot save without optimzier")
    state = {
        "global_step":
            global_step,
        # DataParallel wrap model in attr `module`.
        "model":
            model.module.state_dict()
            if hasattr(model, "module") else model.state_dict(),
        "optim":
            optim.state_dict(),
    }
    save_path = os.path.join(pkg_dir, _file_at_step(global_step, name))
    best_path = os.path.join(pkg_dir, _file_best(name))
    torch.save(state, save_path)
    print("[Checkpoint]: save to {} successfully".format(save_path))

    if is_best:
        copyfile(save_path, best_path)
    if max_checkpoints is not None:
        history = []
        for file_name in os.listdir(pkg_dir):
            if re.search("save_{}_\d*k\d*\.pkg".format(name), file_name):
                digits = file_name.replace("save_{}_".format(name),
                                           "").replace(".pkg", "").split("k")
                number = int(digits[0]) * 1000 + int(digits[1])
                history.append(number)
        history.sort()
        while len(history) > max_checkpoints:
            path = os.path.join(pkg_dir, _file_at_step(history[0], name))
            print("[Checkpoint]: remove {} to keep {} checkpoints".format(
                path, max_checkpoints))
            if os.path.exists(path):
                os.remove(path)
            history.pop(0)


def load(step_or_path, model, name, optim=None, pkg_dir="", device=None):
    step = step_or_path
    save_path = None
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step, name))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best(name))
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint")
        return

    # begin to load
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    model.load_state_dict(state["model"])
    if optim is not None:
        optim.load_state_dict(state["optim"])

    print("[Checkpoint]: Load {} successfully".format(save_path))
    return global_step
