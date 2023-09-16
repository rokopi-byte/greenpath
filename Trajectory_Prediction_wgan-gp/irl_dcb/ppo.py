import torch
import torch.optim as optim
from torch.distributions import Categorical
from .utils import get_walls_loss


class PPO():
    def __init__(self,
                 policy,
                 lr,
                 betas,
                 clip_param,
                 num_epoch,
                 batch_size,
                 value_coef=1.,
                 entropy_coef=0.1,
                 wall_penalty=0.1):

        self.policy = policy
        self.clip_param = clip_param
        self.num_epoch = num_epoch
        self.minibatch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.wall_penalty = wall_penalty
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=lr,
                                    betas=betas)
        self.value_loss_fun = torch.nn.SmoothL1Loss()

    def evaluate_actions(self, obs_batch, actions_batch, fogs):

        probs, values = self.policy(*obs_batch)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_batch)

        return values, log_probs, dist.entropy().mean()

    def update(self, rollouts):
        avg_loss = 0
        entropy_loss = 0
        value_loss = 0
        action_loss = 0
        walls_loss = 0

        for e in range(self.num_epoch):
            data_generator = rollouts.get_generator(self.minibatch_size)

            value_loss_epoch = 0
            action_loss_epoch = 0
            dist_entropy_epoch = 0
            loss_epoch = 0
            wall_loss_epoch = 0

            for i, sample in enumerate(data_generator):
                self.optimizer.zero_grad()
                obs_batch, actions_batch, return_batch, old_action_log_probs_batch, adv_targ, pen, fogs = sample

                fogs = fogs

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.evaluate_actions(
                    obs_batch, actions_batch, fogs)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param)
                surr2 *= adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.value_loss_fun(
                    return_batch, values.squeeze()) * self.value_coef

                #walls_pen = get_walls_loss(actions_batch, obs_batch[0], (47, 47), self.wall_penalty)

                entropy_loss = -dist_entropy * self.entropy_coef
                #pen_loss = pen.mean() * self.wall_penalty
                loss = value_loss + action_loss + entropy_loss #+ pen_loss
                loss.backward()

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += entropy_loss.item()
                loss_epoch += loss.item()
                wall_loss_epoch += pen.mean()

            value_loss_epoch /= i + 1
            action_loss_epoch /= i + 1
            dist_entropy_epoch /= i + 1
            wall_loss_epoch /= i + 1
            loss_epoch /= i + 1

            if e % 4 == 0:
                print('[{}/{}]: VLoss: {:.3f}, PLoss: {:.3f}, WPen: {:.3f}, loss: {:.3f}'.
                      format(e + 1, self.num_epoch, value_loss_epoch,
                             action_loss_epoch, wall_loss_epoch, loss_epoch))
            avg_loss += loss_epoch
            entropy_loss += dist_entropy_epoch
            value_loss += value_loss_epoch
            action_loss += action_loss_epoch
            walls_loss += wall_loss_epoch

        avg_loss /= self.num_epoch
        value_loss /= self.num_epoch
        action_loss /= self.num_epoch
        entropy_loss /= self.num_epoch
        walls_loss /= self.num_epoch

        return avg_loss, value_loss, action_loss, entropy_loss, walls_loss
