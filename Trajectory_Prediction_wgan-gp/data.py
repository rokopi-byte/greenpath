import torch
from . import utils
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import scipy.ndimage as filters
from os.path import join
import warnings
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_path(data, LR_dir, HR_dir, pa, catIds, annos):
    img_name, cat_name, fixs, action, last_point = data
    feat_name = img_name[:-3] + 'pth.tar'
    lr_path = join(LR_dir, cat_name.replace(' ', '_'), feat_name)
    hr_path = join(HR_dir, cat_name.replace(' ', '_'), feat_name)
    state = torch.load(lr_path)
    hr = torch.load(hr_path)

    '''
    fog_of_war = torch.zeros((1, state.size(-1), state.size(-2)))
    fog_of_war_full = torch.ones((1, hr.size(-1), hr.size(-2)))

    hr = torch.cat((fog_of_war_full, hr), dim=0)
    state = torch.cat((fog_of_war, state), dim=0)
    '''
    # construct DCB
    remap_ratio_w = round(pa.im_w[img_name] / float(hr.size(-1)))
    remap_ratio_h = round(pa.im_h[img_name] / float(hr.size(-2)))
    history_map = torch.zeros((hr.size(-2), hr.size(-1)))
    mask = torch.zeros((hr.size(-3), hr.size(-2), hr.size(-1)))
    px = 0
    py = 0
    init_fix = fixs[-1]
    x_init, y_init = int(px / remap_ratio_w), int(py / remap_ratio_h)

    action_init = int(pa.patch_num[0] * y_init + x_init)
    for i in range(len(fixs)):
        px, py = fixs[i]
        px, py = int(px / remap_ratio_w), int(py / remap_ratio_h)
        mask = utils.foveal2mask(px, py, pa.fovea_radius, hr.size(-2),
                                 hr.size(-1))
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
        state = (1 - mask) * state + mask * hr
        history_map[py, px] = 1

    fog_of_war = torch.zeros((hr.size(-2), hr.size(-1)))
    fog_of_war = mask[0] == 0
    if pa.IOR_size > 0:
        fog_of_war[max(py - pa.IOR_size, 0):py + pa.IOR_size + 1,
        max(px - pa.IOR_size, 0):px + pa.IOR_size + 1] = 1
    else:
        fog_of_war[py, px] = 1

    fog_of_war = fog_of_war.view(1, state.size(-2), state.size(-1)).to(torch.float32).to(state.device)
    history_map_w = history_map.view(1, history_map.size(-2), history_map.size(-1))

    # for last point
    #px, py = int(last_point[0]), int(last_point[1])
    # mask = utils.foveal2mask(px, py, self.pa.fovea_radius, hr.size(-2),
    #                         hr.size(-1))
    # mask = torch.from_numpy(mask)
    # mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
    # state = (1 - mask) * state + mask * hr

    # position_map = utils.create_action_matrix_single(px, py, state.size(-2), state.size(-1))\
    #    .view(1, state.size(-2), state.size(-1)).to(state.device)
    state = torch.cat([state, fog_of_war], dim=0)
    state = torch.cat([state, history_map_w], dim=0)

    # utils.show_states(state, action=(action % 47, action // 47))
    #utils.show_states(state, action=(action % 47, action // 47), chan=28, name="fog")

    # utils.show_states(state, action=(action % 47, action // 47), name="fog", chan=28)

    # create labels
    imgId = cat_name + '_' + img_name
    coding = utils.multi_hot_coding(annos[imgId], pa.patch_size[img_name],
                                    pa.patch_num)
    coding = torch.from_numpy(coding / coding.sum()).view(1, -1)

    ret = {
        "task_id": catIds[cat_name],
        "true_state": state,
        "true_action": torch.tensor([action], dtype=torch.long),
        'label_coding': coding,
        'history_map': history_map,
        'last_point': (px * remap_ratio_w, py * remap_ratio_h),
        'img_name': img_name,
        'task_name': cat_name,
        'init_fix': init_fix,
        'action_init': action_init
    }
    return ret


class RolloutStorage_New(object):
    def __init__(self, trajs_all, shuffle=True):
        self.obs_fovs = torch.cat([traj['state'] for traj in trajs_all])
        self.actions = torch.cat([traj['action'] for traj in trajs_all])
        self.tids = torch.cat([traj['task_id'] for traj in trajs_all])
        self.lprobs = torch.cat([traj['log_prob'] for traj in trajs_all])
        self.returns = torch.cat([traj['return']
                                  for traj in trajs_all]).view(-1)
        advs = torch.stack([traj['advantage'] for traj in trajs_all])
        # print(advs.size())
        self.advs = (advs -
                     advs.mean()) / advs.std()  # normalize for stability
        self.advs = self.advs.view(-1)

        self.sample_num = self.obs_fovs.size(0)
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        perm = torch.randperm(
            self.sample_num) if self.shuffle else torch.arange(self.sample_num)
        for start_ind in range(0, self.sample_num, minibatch_size):
            ind = perm[start_ind:start_ind + minibatch_size]
            obs_fov_batch = self.obs_fovs[ind]
            actions_batch = self.actions[ind]
            tids_batch = self.tids[ind]
            return_batch = self.returns[ind]
            log_probs_batch = self.lprobs[ind]
            advantage_batch = self.advs[ind]

            yield (
                      obs_fov_batch, tids_batch
                  ), actions_batch, return_batch, log_probs_batch, advantage_batch


class LHF_IRL(Dataset):
    """
    Image data for training generator
    """

    def __init__(self, DCB_HR_dir, DCB_LR_dir, initial_fix, img_info, annos,
                 pa, catIds, real_dataset=None, chance_of_real=0.1, use_bc = False):
        self.img_info = img_info
        self.annos = annos
        self.pa = pa
        self.initial_fix = initial_fix
        self.catIds = catIds
        self.LR_dir = DCB_LR_dir
        self.HR_dir = DCB_HR_dir

        self.chance_of_real = chance_of_real

        if self.chance_of_real > 0.0 or use_bc:
            self.real_dataset = real_dataset
        else:
            self.real_dataset = None

        self.backup_chance = None
        print("Len", len(self.img_info))

    def change_chance_of_real(self, value):
        if self.backup_chance is None:
            self.backup_chance = self.chance_of_real
        self.chance_of_real = value

    def reset_chance_of_real(self):
        self.chance_of_real = self.backup_chance

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        use_real = False
        if self.real_dataset is not None:
            num = random.random()
            if num < self.chance_of_real:
                use_real = True
        if not use_real:
            vec = self.img_info[idx].split('_')
            # cat_name, img_name = self.img_info[idx].split('_')
            img_name = vec[-1]
            cat_name = self.img_info[idx].replace("_" + img_name, "")
            feat_name = img_name[:-3] + 'pth.tar'
            lr_path = join(self.LR_dir, cat_name.replace(' ', '_'), feat_name)
            hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
            lr = torch.load(lr_path)
            hr = torch.load(hr_path)
            imgId = cat_name + '_' + img_name

            fog_of_war = torch.zeros(1, lr.size(-2), lr.size(-1)).to(lr.device)
            lr = torch.cat([lr, fog_of_war], dim=0)
            hr = torch.cat([hr, fog_of_war], dim=0)

            # update state with initial fixation
            init_fix = self.initial_fix[imgId]
            # px, py = init_fix
            # px, py = px * lr.size(-1), py * lr.size(-2)
            '''
            fog_of_war = torch.zeros((1, lr.size(-1), lr.size(-2)))
            fog_of_war_full = torch.ones((1, hr.size(-1), hr.size(-2)))
    
            lr = torch.cat((fog_of_war, lr), dim=0)
            hr = torch.cat((fog_of_war_full, hr), dim=0)
            '''
            # mask = utils.foveal2mask(px, py, self.pa.fovea_radius, hr.size(-2),
            #                         hr.size(-1))
            # mask = torch.from_numpy(mask)
            # mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
            # lr = (1 - mask) * lr + mask * hr

            # history fixation map
            history_map = torch.zeros((hr.size(-2), hr.size(-1)))
            history_map_w = history_map.view(1, history_map.size(-2), history_map.size(-1))
            # history_map = (1 - mask[0]) * history_map + mask[0] * 1

            lr = torch.cat([lr, history_map_w], dim=0)
            hr = torch.cat([hr, history_map_w], dim=0)

            # action mask
            action_mask = torch.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                      dtype=torch.uint8)
            # px, py = init_fix
            # px, py = int(px * self.pa.patch_num[0]), int(py * self.pa.patch_num[1])
            # print(px, py)
            # action_mask[py - self.pa.IOR_size:py + self.pa.IOR_size + 1, px -
            #            self.pa.IOR_size:px + self.pa.IOR_size + 1] = 1

            # target location label
            coding = utils.multi_hot_coding(self.annos[imgId], self.pa.patch_size[img_name],
                                            self.pa.patch_num)
            coding = torch.from_numpy(coding).view(1, -1)
            task_id = self.catIds[cat_name]
            true_action, action_init = None, None
        else:
            ret = extract_path(self.real_dataset[idx], self.LR_dir, self.HR_dir, self.pa, self.catIds, self.annos)
            task_id = ret["task_id"]
            img_name = ret["img_name"]
            cat_name = ret["task_name"]
            feat_name = img_name[:-3] + 'pth.tar'
            lr = ret["true_state"]
            true_action = ret["true_action"]
            hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
            hr = torch.load(hr_path)
            fog_of_war = torch.zeros(1, hr.size(-2), hr.size(-1)).to(lr.device)
            hr = torch.cat([hr, fog_of_war], dim=0)

            history_map = torch.zeros((hr.size(-2), hr.size(-1)))
            history_map_w = history_map.view(1, history_map.size(-2), history_map.size(-1))
            hr = torch.cat([hr, history_map_w], dim=0)
            init_fix = ret["init_fix"]
            coding = ret["label_coding"]
            action_init = ret["action_init"]
            action_mask = torch.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                      dtype=torch.uint8)
        return_value = {
            'task_id': task_id,
            'img_name': img_name,
            'cat_name': cat_name,
            'lr_feats': lr,
            'hr_feats': hr,
            'real': use_real,
            'history_map': history_map,
            'init_fix': torch.FloatTensor(init_fix),
            'label_coding': coding,
            'action_mask': action_mask
        }
        if self.chance_of_real == 1.0:
            return_value['true_action'] = true_action
            return_value['action_init'] = action_init
        return return_value


class LHF_Human_Gaze(Dataset):
    """
    Human gaze data for training discriminator
    """

    def __init__(self,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 fix_labels,
                 annos,
                 pa,
                 catIds,
                 blur_action=False, abs=None):
        self.pa = pa
        self.fix_labels = fix_labels
        self.annos = annos
        self.catIds = catIds
        self.LR_dir = DCB_LR_dir
        self.HR_dir = DCB_HR_dir
        self.blur_action = blur_action

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        # load low- and high-res beliefs
        img_name, cat_name, fixs, action, last_point = self.fix_labels[idx]
        ret = extract_path(self.fix_labels[idx], self.LR_dir, self.HR_dir, self.pa, self.catIds, self.annos)
        # blur action maps for evaluation
        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            action_map[action] = 1
            action_map = action_map.reshape(self.pa.patch_num[1], -1)
            action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['action_map'] = action_map
        return ret


class RolloutStorage(object):
    def __init__(self, trajs_all, shuffle=True, norm_adv=False):
        self.obs_fovs = torch.cat([traj["curr_states"] for traj in trajs_all])
        self.actions = torch.cat([traj["actions"] for traj in trajs_all])
        self.lprobs = torch.cat([traj['log_probs'] for traj in trajs_all])
        self.tids = torch.cat([traj['task_id'] for traj in trajs_all])
        self.returns = torch.cat([traj['acc_rewards']
                                  for traj in trajs_all]).view(-1)
        self.fogs = torch.cat([traj['fogs'] for traj in trajs_all])
        self.penalties = torch.cat([traj['penalties'] for traj in trajs_all]).view(-1)
        self.advs = torch.cat([traj['advantages']
                               for traj in trajs_all]).view(-1)
        if norm_adv:
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() +
                                                          1e-8)

        self.sample_num = self.obs_fovs.size(0)
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        minibatch_size = min(self.sample_num, minibatch_size)
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               minibatch_size,
                               drop_last=True)
        for ind in sampler:
            obs_fov_batch = self.obs_fovs[ind]
            actions_batch = self.actions[ind]
            tids_batch = self.tids[ind]
            return_batch = self.returns[ind]
            log_probs_batch = self.lprobs[ind]
            advantage_batch = self.advs[ind]
            penalties = self.penalties[ind]
            fogs = self.fogs[ind]

            yield (
                      obs_fov_batch, tids_batch
                  ), actions_batch, return_batch, log_probs_batch, advantage_batch, penalties, fogs


class FakeDataRollout(object):
    def __init__(self, trajs_all, minibatch_size, shuffle=True):
        self.GS = torch.cat([traj['curr_states'] for traj in trajs_all])
        self.GA = torch.cat([traj['actions']
                             for traj in trajs_all]).unsqueeze(1)
        self.tids = torch.cat([traj['task_id'] for traj in trajs_all])
        self.GP = torch.exp(
            torch.cat([traj["log_probs"] for traj in trajs_all])).unsqueeze(1)
        # self.GIOR = torch.cat([traj["IORs"]
        #                        for traj in trajs_all]).unsqueeze(1)

        self.sample_num = self.GS.size(0)
        self.shuffle = shuffle
        self.batch_size = min(minibatch_size, self.sample_num)

    def __len__(self):
        return int(self.sample_num // self.batch_size)

    def get_generator(self):
        # sampler = BatchSampler(SequentialSampler(range(self.sample_num)),
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               self.batch_size,
                               drop_last=True)
        for ind in sampler:
            GS_batch = self.GS[ind]
            tid_batch = self.tids[ind]
            GA_batch = self.GA[ind]
            GP_batch = self.GP[ind]
            # GIOR_batch = self.GIOR[ind]

            yield GS_batch, GA_batch, GP_batch, tid_batch
