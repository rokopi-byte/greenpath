import random

import torch
from .utils import foveal2mask, show_states, create_action_matrix, bresenham
import warnings
import math
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)


class IRL_Env4LHF:
    """
    Environment for low- and high-res DCB under 
    inverse reinforcement learning
    """

    def __init__(self,
                 pa,
                 max_step,
                 mask_size,
                 status_update_mtd,
                 device,
                 inhibit_return=False,
                 init_mtd='first'):
        self.pa = pa
        self.init = init_mtd
        self.max_step = max_step + 1  # one more step to hold the initial step
        self.inhibit_return = inhibit_return
        self.mask_size = mask_size
        self.status_update_mtd = status_update_mtd
        self.device = device
        self.first_fixation = []
        self.action_mask = None
        self.fog_of_war = None
        self.states = None

    def observe(self, accumulate=True):
        if self.step_id >= 0:
            # update state with high-res feature
            # remap_ratio = self.pa.patch_num[0] / float(self.states.size(-1))
            lastest_fixation_on_feats = self.fixations[:, self.step_id].to(dtype=torch.float32) # / remap_ratio
            px = lastest_fixation_on_feats[:, 0]
            py = lastest_fixation_on_feats[:, 1]

            # self.npfog1 = self.states.detach().cpu().numpy()
            masks = []
            masks_simple = []
            for i in range(self.batch_size):
                mask = foveal2mask(px[i].item(), py[i].item(),
                                   self.pa.fovea_radius, self.states.size(-2),
                                   self.states.size(-1))
                mask = torch.from_numpy(mask).to(self.device)
                masks_simple.append(mask)
                mask = mask.unsqueeze(0).repeat(self.states.size(1), 1, 1)
                masks.append(mask)
                if px[i].item() < self.states.size(-2) and py[i].item() < self.states.size(-1):
                    self.history_map[i, py[i].to(torch.long), px[i].to(torch.long)] = 1
            masks = torch.stack(masks)
            masks_simple = torch.stack(masks_simple)
            if accumulate:
                self.states = (1 - masks) * self.states + masks * self.hr_feats
            else:
                self.states = (1 - masks) * self.lr_feats + masks * self.hr_feats

            #position_map = create_action_matrix(px, py, self.states.size(-2), self.states.size(-1))

             # = (1 - masks[:, 0]) * self.history_map + masks[:, 0]
            self.fog_of_war = self.fog_of_war.view(self.fog_of_war.size(0), self.pa.patch_num[1], -1)
            self.action_mask = self.action_mask.view(self.action_mask.size(0), self.pa.patch_num[1], -1)

            self.fog_of_war = masks_simple == 0

            self.states[:, -2] = self.fog_of_war + self.action_mask
            self.states[:, -1] = self.history_map.clone()
            # self.npfog2 = self.states.detach().cpu().numpy()
            self.fog_of_war = self.fog_of_war.view(self.fog_of_war.size(0), -1)

            self.action_mask = self.action_mask.view(self.action_mask.size(0), -1)

        ext_states = self.states.clone()

        return ext_states

    def get_reward(self, prob_old, prob_new):
        return torch.zeros(self.batch_size, device=self.device)

    def return_penalty_action(self, act_batch):
        # torch.set_printoptions(edgeitems=100)
        py, px = act_batch // self.pa.patch_num[
            0], act_batch % self.pa.patch_num[0]
        penalties_tot = torch.zeros(self.batch_size, 1).to(act_batch.device)
        self.fog_of_war = self.fog_of_war.view(self.fog_of_war.size(0), self.pa.patch_num[1], -1).to(torch.long)
        self.action_mask = self.action_mask.view(self.fog_of_war.size(0), self.pa.patch_num[1], -1)

        #np_fog_of_war = self.fog_of_war.cpu().numpy()
        for i in range(self.batch_size):
            last_px = self.fixations[i, self.step_id, 0].to(torch.float32)
            last_py = self.fixations[i, self.step_id, 1].to(torch.float32)

            if self.step_id > 0:
                last_px_1 = self.fixations[i, self.step_id-1, 0].to(torch.float32)
                last_py_1 = self.fixations[i, self.step_id-1, 1].to(torch.float32)
                # delta_x = px[i] - last_px_1
                # delta_y = last_py_1 -  py[i]
                delta_x = last_px - last_px_1
                delta_y = last_py_1 - last_py
                theta_radians_1 = torch.atan2(delta_y, delta_x) * (180.0 / math.pi)

                delta_x = px[i] - last_px
                delta_y = last_py - py[i]
                theta_radians = torch.atan2(delta_y, delta_x) * (180.0 / math.pi)

                angle = torch.abs(theta_radians_1 - theta_radians)
                if angle > 180.0:
                    angle = 360.0 - angle

                if angle > 70:
                    penalties_tot[i] += (angle - 70)

            if self.fog_of_war[i, py[i], px[i]] == 1:
                penalty = torch.sqrt((px[i].to(torch.float32) - last_px)**2 +
                                     (py[i].to(torch.float32) - last_py)**2)
                penalties_tot[i] += penalty - self.pa.fovea_radius
            if self.action_mask[i, py[i], px[i]] == 1:
                distance = torch.sqrt((px[i].to(torch.float32) - last_px)**2 +
                                      (py[i].to(torch.float32) - last_py)**2)
                penalties_tot[i] += (self.pa.IOR_size - distance)*self.pa.fovea_radius
            if self.step_id > 1 and self.states[i, 25, py[i], px[i]] == 0:
                distance = torch.sqrt((px[i].to(torch.float32) - last_px) ** 2 +
                                          (py[i].to(torch.float32) - last_py) ** 2)
                penalties_tot[i] += distance*100
            #print("_____",last_px.item(), last_py.item())
            if self.step_id > 1:
                for pnt in bresenham((last_px.item(), last_py.item()), (px[i].item(), py[i].item())):
                    bs_x, bs_y = pnt

                    if (bs_x == last_px.item() and bs_y == last_py.item()) or \
                            (bs_x == px[i].item() and bs_y == py[i].item()):
                        continue
                    distance = torch.sqrt((px[i].to(torch.float32) - bs_x) ** 2 +
                                          (py[i].to(torch.float32) - bs_y) ** 2)
                    #print(bs_x, bs_y)
                    if bs_x >= self.pa.patch_num[0]:
                        bs_x = self.pa.patch_num[0]-1
                    if bs_y >= self.pa.patch_num[1]:
                        bs_y = self.pa.patch_num[1] - 1
                    if self.states[i, 25, bs_y, bs_x] == 0:
                        penalties_tot[i] += distance*0.1
            #print("_____", px[i].item(), py[i].item())
            if self.history_map[i, py[i], px[i]] == 1:
                distance = torch.sqrt((px[i].to(torch.float32) - last_px)**2 +
                                      (py[i].to(torch.float32) - last_py)**2)
                penalties_tot[i] += torch.abs(self.pa.fovea_radius - distance)
            if penalties_tot[i] < 0:
                penalties_tot[i] = 0

        self.fog_of_war = self.fog_of_war.view(self.fog_of_war.size(0), -1)
        self.action_mask = self.action_mask.view(self.fog_of_war.size(0), -1)
        return penalties_tot

    def set_first_fixations(self, ff):
        self.first_fixation = ff

    def step(self, act_batch):
        self.step_id += 1
        assert self.step_id < self.max_step, "Error: Exceeding maximum step!"

        # update fixation
        py, px = act_batch // self.pa.patch_num[
            0], act_batch % self.pa.patch_num[0]

        self.fixations[:, self.step_id, 1] = py
        self.fixations[:, self.step_id, 0] = px

        # update action mask
        if self.inhibit_return:
            self.action_mask[:] = 0
            self.previous_position[:, act_batch] = 1
            if self.mask_size == 0:
                self.action_mask[:, act_batch] = 1

            else:
                bs = self.action_mask.size(0)
                px, py = px.to(dtype=torch.long), py.to(dtype=torch.long)
                self.action_mask = self.action_mask.view(
                    bs, self.pa.patch_num[1], -1)
                #self.previous_position = self.previous_position.view(
                #    bs, self.pa.patch_num[1], -1)
                #for i in range(bs):
                #    self.previous_position[i,
                #                     max(py[i] - self.mask_size, 0):py[i] +
                #                     self.mask_size + 1,
                #                     max(px[i] - self.mask_size, 0):px[i] +
                #                     self.mask_size + 1] = 1
                #self.previous_position = self.previous_position.view(bs, -1)

                for i in range(bs):
                    self.action_mask[i,
                                     max(py[i] - self.mask_size, 0):py[i] +
                                     self.mask_size + 1,
                                     max(px[i] - self.mask_size, 0):px[i] +
                                     self.mask_size + 1] = 1

                self.action_mask = self.action_mask.view(bs, -1)

        # show_states(self.states[0], "step", (px[0].cpu().numpy(), py[0].cpu().numpy()))
        # show_states(self.states[0], "step", (px[0].cpu().numpy(), py[0].cpu().numpy()), -2)
        obs = self.observe()
        self.status_update(act_batch)

        #print(self.status)

        return obs, self.status

    def status_update(self, act_batch):
        if self.status_update_mtd == 'SOT':  # stop on target
            done = self.label_coding[torch.arange(self.batch_size
                                                  ), 0, act_batch]
        else:
            raise NotImplementedError

        done[self.status > 0] = 2
        self.status = done.to(torch.uint8)

    def check_task(self, i, act):
        return self.label_coding[i, 0, act]

    def step_back(self):
        self.fixations[:, self.step_id] = 0
        self.step_id -= 1

    def reset(self, force_init=False):
        self.step_id = 0  # step id of the environment
        self.fixations = torch.zeros((self.batch_size, self.max_step, 2),
                                     dtype=torch.long,
                                     device=self.device)
        self.status = torch.zeros(self.batch_size,
                                  dtype=torch.uint8,
                                  device=self.device)
        self.history_map = self.history_map_or.clone()
        self.states = self.lr_feats.clone()

        self.action_mask = self.init_action_mask.clone()
        self.action_mask[:] = 0
        self.previous_position = self.action_mask.clone()
        self.previous_position = self.previous_position.view(self.batch_size, -1)

        if self.init == 'center' and not force_init:
            self.fixations[:, 0] = torch.tensor(
                [[self.pa.patch_num[0] / 2, self.pa.patch_num[1] / 2]],
                dtype=torch.long,
                device=self.device)
            bs = self.action_mask.size(0)
            self.action_mask = self.action_mask.view(bs, self.pa.patch_num[1],
                                                     -1)
            px, py = int(self.pa.patch_num[0] / 2), int(self.pa.patch_num[1] /
                                                        2)
            self.action_mask[:, py - self.mask_size:py + self.mask_size +
                             1, px - self.mask_size:px + self.mask_size +
                             1] = 1

            self.action_mask = self.action_mask.view(bs, -1)
            self.fog_of_war = torch.zeros(self.action_mask.size(), device=self.device, dtype=torch.uint8)
        elif self.init == 'real' and not force_init:
            self.action_mask = self.action_mask.view(self.batch_size, self.pa.patch_num[1],
                                                     -1)
            for i in range(self.batch_size):
                remap_ratio_w = round(self.pa.im_w[self.img_names[i]] / float(self.pa.patch_num[0]))
                remap_ratio_h = round(self.pa.im_h[self.img_names[i]] / float(self.pa.patch_num[1]))
                end = False
                while not end:
                    try:
                        init = random.choice(self.first_fixation[self.img_names[i] + "_" + self.cat_names[i]])
                    except KeyError as e:
                        init = random.choice(self.first_fixation[self.img_names[i]])
                    if self.is_real_state[i]:
                        px, py = self.init_fix[i]
                        px = int(px / remap_ratio_w)
                        py = int(py / remap_ratio_h)
                        end = True
                    else:
                        px, py = init
                        px = int(px / remap_ratio_w)
                        py = int(py / remap_ratio_h)

                        if px > 46:
                            px = 46
                        if py > 46:
                            py = 46
                        value = self.check_task(i,  int(self.pa.patch_num[0] * py + px))
                        end = (value < 1e-08)

                self.fixations[i, 0] = torch.tensor(
                    [[px, py]],
                    dtype=torch.long,
                    device=self.device)

                self.action_mask[i, max(0, py - self.mask_size):py + self.mask_size + 1,
                                    max(0, px - self.mask_size):px + self.mask_size + 1] = 1

            self.action_mask = self.action_mask.view(self.batch_size, -1)
            self.fog_of_war = torch.zeros(self.action_mask.size(), device=self.device, dtype=torch.uint8)
        elif self.init == 'manual' and not force_init:
            self.fixations[:, 0, 0] = self.init_fix[:, 0]
            self.fixations[:, 0, 1] = self.init_fix[:, 1]
        elif self.init == 'random' and not force_init:
            bs = self.action_mask.size(0)
            self.action_mask = self.action_mask.view(self.batch_size, self.pa.patch_num[1],
                                                     -1)
            for i in range(self.batch_size):
                end = False
                while not end:
                    init = torch.rand(2)
                    px = int(init[0] * self.pa.patch_num[0])
                    py = int(init[1] * self.pa.patch_num[1])
                    end = (self.check_task(i, int(self.pa.patch_num[0] * py + px)) < 1e-08)

                self.fixations[i, 0] = torch.tensor(
                    [[px, py]],
                    dtype=torch.long,
                    device=self.device)

                self.action_mask[i, max(0, py - self.mask_size):py + self.mask_size + 1,
                max(0, px - self.mask_size):px + self.mask_size + 1] = 1

            self.action_mask = self.action_mask.view(bs, -1)
            self.fog_of_war = torch.zeros(self.action_mask.size(), device=self.device, dtype=torch.uint8)
        elif self.init == 'first' or force_init:
            bs = self.action_mask.size(0)
            self.action_mask = self.action_mask.view(bs, self.pa.patch_num[1],
                                                     -1)
            for i in range(self.batch_size):
                px = int(self.init_fix[i][0] * self.pa.patch_num[0])
                py = int(self.init_fix[i][1] * self.pa.patch_num[1])

                if px > 46:
                    px = 46
                if py > 46:
                    py = 46

                self.fixations[i, 0] = torch.tensor(
                    [[px, py]],
                    dtype=torch.long,
                    device=self.device)

                self.action_mask[i, py - self.pa.IOR_size:py + self.pa.IOR_size+1,
                px - self.pa.IOR_size:px + self.pa.IOR_size+1] = 1

            self.action_mask = self.action_mask.view(bs, -1)
            self.fog_of_war = torch.zeros(self.action_mask.size(), device=self.device, dtype=torch.uint8)

        else:
            raise NotImplementedError

    def set_data(self, data):
        self.label_coding = data['label_coding'].to(self.device)
        self.np_label =  self.label_coding.view(self.label_coding.size(0), 47, -1).cpu().numpy()
        self.img_names = data['img_name']
        self.cat_names = data['cat_name']
        self.init_fix = data['init_fix'].to(self.device)
        self.init_action_mask = data['action_mask'].to(self.device)
        self.history_map_or = data['history_map'].to(self.device)
        self.history_map = self.history_map_or.clone()
        self.task_ids = data['task_id'].to(self.device)
        self.lr_feats = data['lr_feats'].to(self.device)
        self.hr_feats = data['hr_feats'].to(self.device)
        self.is_real_state = data['real']
        self.last_position = torch.zeros(self.lr_feats.size(0), 2)
        self.batch_size = self.hr_feats.size(0)


        empty_position = torch.zeros(self.lr_feats.size(0), 1, self.lr_feats.size(-2),
                                     self.lr_feats.size(-1)).to(self.device)

        if self.inhibit_return:
            self.action_mask = data['action_mask'].to(self.device).view(
                self.batch_size, -1)
        else:
            self.action_mask = torch.zeros(self.batch_size,
                                           self.pa.patch_count,
                                           dtype=torch.uint8)
        self.fog_of_war = torch.zeros(self.action_mask.size(), device=self.device, dtype=torch.uint8)
        self.reset()
