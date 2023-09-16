import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .ppo import PPO
from .gail import GAIL
from . import utils
from .data import RolloutStorage, FakeDataRollout


class Trainer(object):
    def __init__(self, model, loaded_step, env, dataset, device, hparams):
        self.global_step = 0
        # setup logger
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        self.log_dir = os.path.join(hparams.Train.log_root, "log_" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoint_every = hparams.Train.checkpoint_every
        self.max_checkpoints = hparams.Train.max_checkpoints

        self.loaded_step = loaded_step
        self.env = env['train']
        self.env_valid = env['valid']
        self.generator = model['gen']
        self.discriminator = model['disc']
        self.bbox_annos = dataset['bbox_annos']
        self.human_mean_cdf = dataset['human_mean_cdf']
        self.device = device

        self.dataset_generator = dataset['img_train']

        # image dataloader
        self.batch_size = hparams.Train.batch_size
        self.train_img_loader = DataLoader(self.dataset_generator,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0)
        self.valid_img_loader = DataLoader(dataset['img_valid'],
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0)

        # human gaze dataloader
        self.train_HG_loader = DataLoader(dataset['gaze_train'],
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=0)

        self.not_walkable = dataset['not_walkable']

        # training parameters
        self.gamma = hparams.Train.gamma
        self.adv_est = hparams.Train.adv_est
        self.tau = hparams.Train.tau
        self.max_traj_len = hparams.Data.max_traj_length
        self.n_epoches = hparams.Train.num_epoch
        self.n_steps = hparams.Train.num_step
        self.n_critic = hparams.Train.num_critic
        self.patch_num = hparams.Data.patch_num
        self.penalty_wall = hparams.Train.penalty_wall
        self.im_w = hparams.Data.im_w
        self.im_h = hparams.Data.im_h
        self.eval_every = hparams.Train.evaluate_every
        self.ppo = PPO(self.generator, hparams.PPO.lr,
                       hparams.Train.adam_betas, hparams.PPO.clip_param,
                       hparams.PPO.num_epoch, hparams.PPO.batch_size,
                       hparams.PPO.value_coef, hparams.PPO.entropy_coef, hparams.PPO.wall_penalty)
        self.gail = GAIL(self.discriminator,
                         hparams.Train.gail_milestones,
                         None,
                         device,
                         lr=hparams.Train.gail_lr,
                         betas=hparams.Train.adam_betas)
        self.writer = SummaryWriter(self.log_dir)
        self.bc_iters = hparams.Train.bc_iters
        if self.bc_iters > 0:
            self.bc_loss = torch.nn.BCELoss()
            self.optimizer_bc = optim.Adam(self.generator.parameters(), lr=hparams.Train.bc_lr)

    def train_bc(self):
        self.dataset_generator.change_chance_of_real(1.0)
        iters = 0
        epoch = 0
        while iters < self.bc_iters:
            print("Epoch:", epoch, "Iteration:", iters)
            epoch += 1
            for i_batch, batch in enumerate(self.train_img_loader):
                if iters > self.bc_iters:
                    break
                self.env.set_data(batch)
                self.env.reset()

                trajs = utils.collect_trajs(self.env, self.generator,
                                            self.patch_num, 1, sample_action=False, is_eval=True)
                act = trajs['probs'].to(self.device).to(torch.float).squeeze()
                act_real = batch['true_action'].to(self.device)
                act_init = batch['action_init'].to(self.device)

                lst = []
                lst_init = []
                label = torch.ones_like(act_real).to(self.device).to(torch.float)
                label_init = torch.zeros_like(act_init).to(self.device).to(torch.float)

                for i in range(len(act)):
                    lst.append(act[i, act_real[i].to(torch.long)])
                    lst_init.append(act[i, act_init[i].to(torch.long)])
                result = torch.stack(lst)
                result_init = torch.stack(lst_init)

                self.optimizer_bc.zero_grad()
                loss = self.bc_loss(result, label)
                loss_init = self.bc_loss(result_init, label_init)  # we force the agent to avoid staying on place
                (loss + loss_init).backward()
                self.optimizer_bc.step()
                if iters % 100 == 0:
                    print("Iteration", iters, "BC Loss:", (loss + loss_init).item())
                iters += 1
        print("Saving bc model...")
        utils.save(global_step=self.global_step,
                   model=self.generator,
                   optim=self.optimizer_bc,
                   name='generator_bc',
                   pkg_dir=self.checkpoints_dir,
                   is_best=True,
                   max_checkpoints=self.max_checkpoints)
        self.optimizer_bc.zero_grad()
        self.optimizer_bc = None
        self.dataset_generator.reset_chance_of_real()

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.global_step = self.loaded_step

        if self.bc_iters > 0:
            print("Starting behavioral cloning initialization...")
            self.train_bc()
            print("Ended behavioural cloning.")

        for i_epoch in range(self.n_epoches):
            print("Epoch {} of {}. Global step n.{}".format(i_epoch, self.n_epoches, self.global_step))
            for i_batch, batch in enumerate(self.train_img_loader):
                # run policy to collect trajectories
                print(
                    "generating state-action pairs to train discriminator...")
                trajs_all = []
                self.env.set_data(batch)
                return_train = 0.0
                smp_num = 0
                while smp_num <= 900:
                    for i_step in range(self.n_steps):
                        with torch.no_grad():
                            self.env.reset()
                            trajs = utils.collect_trajs(self.env, self.generator,
                                                        self.patch_num,
                                                        self.max_traj_len)
                            trajs_all.extend(trajs)
                    smp_num = np.sum(list(map(lambda x: x['length'], trajs_all)))
                print("[{} {}] Collected {} state-action pairs".format(
                    i_epoch, i_batch, smp_num))

                # train discriminator (reward and value function)
                print("updating discriminator (step={})...".format(
                    self.gail.update_counter))

                fake_data = FakeDataRollout(trajs_all, self.batch_size)
                D_loss, D_real, D_fake = self.gail.update(
                    self.train_HG_loader, fake_data)

                self.writer.add_scalar("discriminator/fake_loss", D_fake,
                                       self.global_step)
                self.writer.add_scalar("discriminator/real_loss", D_real,
                                       self.global_step)
                self.writer.add_scalar("discriminator/learning_rate", self.gail.optimizer.param_groups[0]['lr'],
                                       self.global_step)

                print("Done updating discriminator!")

                # evaluate generator/policy
                if self.global_step > 0 and \
                        self.global_step % self.eval_every == 0:
                    print("evaluating policy...")

                    # generating scanpaths
                    all_actions = []
                    for i_sample in range(2):
                        for batch in self.valid_img_loader:
                            self.env_valid.set_data(batch)
                            img_names_batch = batch['img_name']
                            cat_names_batch = batch['cat_name']
                            with torch.no_grad():
                                self.env_valid.reset()
                                trajs = utils.collect_trajs(self.env_valid,
                                                            self.generator,
                                                            self.patch_num,
                                                            self.max_traj_len,
                                                            is_eval=True,
                                                            sample_action=True)
                                all_actions.extend([
                                    (cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i], trajs['init'][i])
                                    for i in range(self.env_valid.batch_size)
                                ])
                    scanpaths = utils.actions2scanpaths(
                        all_actions, self.patch_num, self.im_w, self.im_h)
                    utils.cutFixOnTarget(scanpaths, self.bbox_annos)

                    # search effiency
                    mean_cdf, _ = utils.compute_search_cdf(
                        scanpaths, self.bbox_annos, self.max_traj_len)
                    self.writer.add_scalar('evaluation/TFP_step3', mean_cdf[3],
                                           self.global_step)
                    self.writer.add_scalar('evaluation/TFP_step7', mean_cdf[7],
                                           self.global_step)
                    self.writer.add_scalar('evaluation/TFP_step10', mean_cdf[10],
                                           self.global_step)

                    # probability mismatch
                    sad = np.sum(np.abs(self.human_mean_cdf - mean_cdf))
                    self.writer.add_scalar('evaluation/prob_mismatch', sad,
                                           self.global_step)
                    print("Mismatch:", sad)
                    print("Human CDF:", self.human_mean_cdf)
                    print("Generated CDF:", mean_cdf)

                # update generator/policy on every n_critic iter
                if i_batch % self.n_critic == 0:
                    print("updating policy...")
                    # update reward and value
                    penalty = 0
                    avg_point_penalty = 0
                    with torch.no_grad():
                        rew_sum = 0
                        for i in range(len(trajs_all)):
                            states = trajs_all[i]["curr_states"]
                            # or_len = len(states)
                            actions = trajs_all[i]["actions"].unsqueeze(1)
                            tids = trajs_all[i]['task_id']
                            '''
                            diff = self.max_traj_len - or_len
                            last_state = trajs_all[i]["last_state"]
                            last_state = last_state.view(1, last_state.size(0), last_state.size(1),
                                                         last_state.size(2))
                            if diff > 0:
                                for j in range(diff):
                                    states = torch.cat([states, last_state], dim=0)

                                    actions = torch.cat([actions, actions[-1].view(1, 1)], dim=0)
                                    tids = torch.cat([tids, tids[-1].view(1)], dim=0)
                            
                            rewards = F.logsigmoid(self.discriminator(states, actions, tids))
                            '''
                            #rewards = self.discriminator(states, actions, tids)
                            rewards = F.logsigmoid(self.discriminator(states, actions, tids)) - \
                                      torch.log(1 - F.sigmoid(self.discriminator(states, actions, tids)) + 1e-08)
                            # diff = self.max_traj_len - trajs_all[i]["length"]
                            # trajs_all[i]["rewards"] = rewards
                            trajs_all[i]["rewards"] = rewards
                            if torch.mean(trajs_all[i]["penalties"]) > 6.0:
                                trajs_all[i]["rewards"] -= trajs_all[i]["penalties"].cuda().view(-1) * self.penalty_wall
                            trajs_all[i]["abs_rewards"] = None  # rewards[or_len:] - trajs_all[i]["penalties"].cuda().view(-1)[-1] * self.penalty_wall

                            # pen_length = (torch.mean(trajs_all[i]['rewards']) * diff) / (trajs_all[i]["length"])

                            # trajs_all[i]["rewards"] += pen_length

                    return_train, avg_penalties = utils.process_trajs(trajs_all,
                                                                      self.gamma,
                                                                      mtd=self.adv_est,
                                                                      tau=self.tau)
                    self.writer.add_scalar("generator/ppo_return",
                                           return_train, self.global_step)
                    self.writer.add_scalar("generator/penalty_wall",
                                           penalty, self.global_step)
                    print('average return = {:.3f}'.format(return_train))
                    print('average traj penalty = {:.3f}'.format(avg_penalties[0].item()))
                    # print('average abs = {:.3f}'.format(avg_abs))
                    # update policy
                    rollouts = RolloutStorage(trajs_all,
                                              shuffle=True,
                                              norm_adv=False)
                    loss, value_loss, action_loss, entropy_loss, walls_loss = self.ppo.update(rollouts)
                    self.writer.add_scalar("generator/ppo_loss", loss,
                                           self.global_step)
                    self.writer.add_scalar("generator/ppo_value_loss", value_loss,
                                           self.global_step)
                    self.writer.add_scalar("generator/ppo_action_loss", action_loss,
                                           self.global_step)
                    self.writer.add_scalar("generator/ppo_entropy_loss", entropy_loss,
                                           self.global_step)
                    self.writer.add_scalar("generator/ppo_walls_loss", walls_loss,
                                           self.global_step)
                    self.writer.add_scalar("generator/lr", self.ppo.optimizer.param_groups[0]['lr'],
                                           self.global_step)

                    print("Done updating policy")

                # checkpoints
                if self.global_step % self.checkpoint_every == 0 and \
                        self.global_step > 0:
                    print("Saving...")
                    utils.save(global_step=self.global_step,
                               model=self.generator,
                               optim=self.ppo.optimizer,
                               name='generator',
                               pkg_dir=self.checkpoints_dir,
                               is_best=True,
                               max_checkpoints=self.max_checkpoints)
                    utils.save(global_step=self.global_step,
                               model=self.discriminator,
                               optim=self.gail.optimizer,
                               name='discriminator',
                               pkg_dir=self.checkpoints_dir,
                               is_best=True,
                               max_checkpoints=self.max_checkpoints)
                    print("Saved!")

                self.global_step += 1

        self.writer.close()
