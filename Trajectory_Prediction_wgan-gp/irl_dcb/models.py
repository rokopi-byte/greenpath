import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import show_states


# conditional version of IRL discriminator (on task) for 20x32
class LHF_Discriminator_Cond(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        super(LHF_Discriminator_Cond, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.padding = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(ch + target_size, 128, 3)
        self.ln1 = nn.LayerNorm([128, 47, 47])
        self.conv2 = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.ln2 = nn.LayerNorm([64, 47, 47])
        self.conv3 = nn.Conv2d(64 + target_size, 32, 3, padding=1)
        self.ln3 = nn.LayerNorm([32, 47, 47])
        self.conv4 = nn.Conv2d(32 + target_size, 1, 1)
        #self.result = nn.Linear(1, 1)

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.LeakyReLU()

        self.task_eye = task_eye

    def get_one_hot(self, tid):
        task_onehot = self.task_eye[tid.to(torch.long)]
        return task_onehot

    def modulate_features(self, feat_maps, tid_onehot):
        """modulat feature maps using task vector"""
        bs, _, h, w = feat_maps.size()
        task_maps = tid_onehot.expand(bs, tid_onehot.size(1), h, w)
        return torch.cat([feat_maps, task_maps], dim=1)

    def forward(self, x, action, tid, ret_mean=False, show=False):
        """ output probability of x being true data"""
        bs, _, h, w = x.size()
        if show:
            for i in range(0, bs):
                show_states(x[i], action=(action[i] % 47, action[i] // 47))
                show_states(x[i], action=(action[i] % 47, action[i] // 47), name="fog", chan=28)
                show_states(x[i], action=(action[i] % 47, action[i] // 47), name="prev", chan=29)

        tid_onehot = self.get_one_hot(tid)
        tid_onehot = tid_onehot.view(bs, tid_onehot.size(1), 1, 1)

        # x_act = x.detach().cpu().numpy()

        x = self.padding(x)
        x = self.modulate_features(x, tid_onehot)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(self.ln1(x))
        if h == 80:
            x = self.max_pool(x)
        x = self.modulate_features(x, tid_onehot)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu(self.ln2(x))
        if h == 80:
            x = self.max_pool(x)
        x = self.modulate_features(x, tid_onehot)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.relu(self.ln3(x))
        x = self.modulate_features(x, tid_onehot)
        x = self.conv4(x)
        #npmap = x.detach().cpu().numpy()
        x = x.view(bs, -1)

        if show:
            print(x.mean())

        if action is None:  # return whole reward map
            return x
        else:
            if ret_mean:
                return x[torch.arange(bs), action.to(torch.long).squeeze()], x.mean()
            else:
                return x[torch.arange(bs), action.to(torch.long).squeeze()]


# conditional version of IRL generator (on task)
class LHF_Policy_Cond_Big(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        super(LHF_Policy_Cond_Big, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.padding = nn.ReplicationPad2d(1)
        self.feat_enc = nn.Conv2d(ch + target_size, 256, 3)
        self.ln0 = nn.LayerNorm([256, 47, 47])

        # actor
        self.actor1 = nn.Conv2d(256 + target_size, 128, 3, padding=1)
        self.ln1 = nn.LayerNorm([128, 47, 47])
        self.actor2 = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.ln2 = nn.LayerNorm([64, 47, 47])
        self.actor3 = nn.Conv2d(64 + target_size, 32, 3, padding=1)
        self.ln3 = nn.LayerNorm([32, 47, 47])
        self.actor4 = nn.Conv2d(32 + target_size, 1, 1)
        
        # critic
        self.critic0 = nn.Conv2d(256 + target_size, 256, 3, padding=1)
        self.ln4 = nn.LayerNorm([256, 47, 47])
        self.critic0_sub = nn.Conv2d(256 + target_size, 256, 3, stride=2)
        self.critic1 = nn.Conv2d(256 + target_size, 512, 3, padding=1)
        self.critic1_sub = nn.Conv2d(512 + target_size, 256, 3, stride=2)
        self.critic2 = nn.Linear(256 + target_size, 64)
        self.critic3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()

        self.task_eye = task_eye

        torch.nn.init.xavier_uniform(self.actor1.weight,  nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.actor2.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.actor3.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.actor4.weight)
        torch.nn.init.xavier_uniform(self.feat_enc.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic0.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic0_sub.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic1.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic1_sub.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic2.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic3.weight)

    def get_one_hot(self, tid):
        task_onehot = self.task_eye[tid]
        return task_onehot

    def modulate_features(self, feat_maps, tid_onehot):
        """modulat feature maps using task vector"""
        bs, _, h, w = feat_maps.size()
        task_maps = tid_onehot.expand(bs, tid_onehot.size(1), h, w)
        return torch.cat([feat_maps, task_maps], dim=1)

    def forward(self, x, tid, act_only=False, show=False):
        """ output the action probability"""
        bs, _, h, w = x.size()
        #x_act = x.detach().cpu().numpy()
        if show:
            show_states(x[0])
        tid_onehot = self.get_one_hot(tid)
        tid_onehot = tid_onehot.view(bs, tid_onehot.size(1), 1, 1)

        x = self.padding(x)
        x = self.modulate_features(x, tid_onehot)
        x = self.feat_enc(x)
        x = self.dropout(x)
        x = self.relu(self.ln0(x))

        x = self.modulate_features(x, tid_onehot)
        act_logits = self.actor1(x)
        act_logits = self.dropout(act_logits)
        act_logits = self.relu(self.ln1(act_logits))

        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor2(act_logits)
        act_logits = self.dropout(act_logits)
        act_logits = self.relu(self.ln2(act_logits))

        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor3(act_logits)
        act_logits = self.dropout(act_logits)
        act_logits = self.relu(self.ln3(act_logits))

        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor4(act_logits)
        #act_logits = self.dropout(act_logits)
        #act_logits = act_logits

        act_logits = act_logits.view(bs, -1)

        torch.autograd.set_detect_anomaly(True)

        #npmap = act_logits.view(bs, 47, 47).detach().cpu().numpy()
        
        act_probs = F.softmax(act_logits, dim=-1)
        #act_np = act_probs.view(bs, 47, 47).detach().cpu().numpy()
        if act_only:
            return act_probs, None
        x = self.critic0(x)

        x = self.relu(self.ln4(x))
        x = self.modulate_features(x, tid_onehot)
        x = self.relu(self.critic0_sub(x))

        x = self.modulate_features(x, tid_onehot)
        x = self.relu(self.critic1(x))

        x = self.modulate_features(x, tid_onehot)
        x = self.relu(self.critic1_sub(x))

        x = x.view(bs, x.size(1), -1).mean(dim=-1)
        list_x = [x, tid_onehot.squeeze()]
        x = torch.cat(list_x, dim=1)
        x = self.relu(self.critic2(x))
        state_values = self.critic3(x)
        #state_values = self.dropout(state_values)
        return act_probs, state_values


class LHF_Policy_Cond_Small(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        super(LHF_Policy_Cond_Small, self).__init__()
        #self.padding = nn.ReplicationPad2d(1)
        self.feat_enc = nn.Conv2d(ch + target_size, 128, 3, padding=1)
        self.ln0 = nn.LayerNorm([128, 47, 47])
        self.bn0 = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(128)

        # actor
        self.actor1 = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.ln1 = nn.LayerNorm([64, 47, 47])
        self.actor2 = nn.Conv2d(64 + target_size, 32, 3, padding=1)
        self.ln2 = nn.LayerNorm([32, 47, 47])
        self.actor3 = nn.Conv2d(32 + target_size, 1, 1)

        # critic
        self.critic0 = nn.Conv2d(128 + target_size, 128, 3, padding=1)
        self.ln3 = nn.LayerNorm([128, 47, 47])
        self.critic0_sub = nn.Conv2d(128 + target_size, 128, 3, stride=2)
        self.critic1 = nn.Conv2d(128 + target_size, 256, 3, padding=1)
        self.critic1_sub = nn.Conv2d(256 + target_size, 256, 3, stride=2)
        self.critic2 = nn.Linear(256 + target_size, 64)
        self.critic3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()

        self.task_eye = task_eye

        torch.nn.init.xavier_uniform(self.actor1.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.actor2.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.actor3.weight)
        #torch.nn.init.xavier_uniform(self.actor4.weight)
        torch.nn.init.xavier_uniform(self.feat_enc.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic0.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic0_sub.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic1.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic1_sub.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic2.weight, nn.init.calculate_gain('leaky_relu', 0.01))
        torch.nn.init.xavier_uniform(self.critic3.weight)

    def get_one_hot(self, tid):
        task_onehot = self.task_eye[tid]
        return task_onehot

    def modulate_features(self, feat_maps, tid_onehot):
        """modulat feature maps using task vector"""
        bs, _, h, w = feat_maps.size()
        task_maps = tid_onehot.expand(bs, tid_onehot.size(1), h, w)
        return torch.cat([feat_maps, task_maps], dim=1)

    def forward(self, x, tid, act_only=False, show=False):
        """ output the action probability"""
        bs, _, h, w = x.size()
       # x_act = x.detach().cpu().numpy()
        if show:
            show_states(x[0])
        tid_onehot = self.get_one_hot(tid)
        tid_onehot = tid_onehot.view(bs, tid_onehot.size(1), 1, 1)

        #x = self.padding(x)
        x = self.modulate_features(x, tid_onehot)
        x = self.feat_enc(x)
        #x = self.dropout(x)
        x = self.relu(self.ln0(x))

        x = self.modulate_features(x, tid_onehot)
        act_logits = self.actor1(x)
        #act_logits = self.dropout(act_logits)
        act_logits = self.relu(self.ln1(act_logits))

        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor2(act_logits)
        #act_logits = self.dropout(act_logits)
        act_logits = self.relu(self.ln2(act_logits))

        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor3(act_logits)
        # act_logits = self.dropout(act_logits)
        # act_logits = act_logits

        act_logits = act_logits.view(bs, -1)

        #torch.autograd.set_detect_anomaly(True)

        # npmap = act_logits.view(bs, 47, 47).detach().cpu().numpy()

        act_probs = F.softmax(act_logits, dim=-1)
        #act_np = act_probs.view(bs, 47, 47).detach().cpu().numpy()
        if act_only:
            return act_probs, None
        x = self.critic0(x)

        x = self.relu(self.ln3(x))
        x = self.modulate_features(x, tid_onehot)
        x = self.relu(self.critic0_sub(x))

        x = self.modulate_features(x, tid_onehot)
        x = self.relu(self.critic1(x))

        x = self.modulate_features(x, tid_onehot)
        x = self.relu(self.critic1_sub(x))

        x = x.view(bs, x.size(1), -1).mean(dim=-1)
        list_x = [x, tid_onehot.squeeze()]
        x = torch.cat(list_x, dim=1)
        x = self.relu(self.critic2(x))
        state_values = self.critic3(x)
        # state_values = self.dropout(state_values)
        return act_probs, state_values