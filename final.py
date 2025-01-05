import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################
# 1. 数据生成 (Algorithm1)
############################

def generate_items_by_algorithm1(N=20, container_size=(100, 100, 100)):
    """
    生成物品列表，每个物品以元组 (l, w, h) 表示尺寸
    """
    items = [tuple(container_size)]  # 初始容器
    while len(items) < N:
        # 按体积从大到小排序
        items.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
        big_item = items.pop(0)
        l, w, h = big_item
        axis_lengths = [l, w, h]
        axis_index = random.choices([0, 1, 2], weights=axis_lengths)[0]
        half_len = axis_lengths[axis_index] * 0.5
        offset = random.uniform(-half_len, half_len)
        split_pt = (axis_lengths[axis_index] / 2) + offset

        if split_pt <= 1 or split_pt >= axis_lengths[axis_index] - 1:
            items.append(big_item)
            continue

        size1 = list(big_item)
        size2 = list(big_item)
        size1[axis_index] = split_pt
        size2[axis_index] = axis_lengths[axis_index] - split_pt

        # 随机旋转
        rot_candidates = [
            (size1[0], size1[1], size1[2]),
            (size1[0], size1[2], size1[1]),
            (size1[1], size1[0], size1[2]),
            (size1[1], size1[2], size1[0]),
            (size1[2], size1[0], size1[1]),
            (size1[2], size1[1], size1[0]),
        ]
        size1 = random.choice(rot_candidates)

        rot_candidates = [
            (size2[0], size2[1], size2[2]),
            (size2[0], size2[2], size2[1]),
            (size2[1], size2[0], size2[2]),
            (size2[1], size2[2], size2[0]),
            (size2[2], size2[0], size2[1]),
            (size2[2], size2[1], size2[0]),
        ]
        size2 = random.choice(rot_candidates)

        items.append(tuple(size1))
        if len(items) < N:
            items.append(tuple(size2))

    return items

def create_dataset(num_samples=500, max_items=20, container_size=(100, 100, 100)):
    """
    创建数据集，每个样本是物品尺寸的列表
    """
    dataset = []
    for _ in range(num_samples):
        items = generate_items_by_algorithm1(N=max_items, container_size=container_size)
        dataset.append(items)  # items 是元组列表
    return dataset

############################
# 2. 环境 (单容器3D BPP)
############################

def check_overlap(pos1, size1, pos2, size2):
    """
    检查两个物体在3D空间中是否重叠
    """
    for i in range(3):
        if pos1[i] + size1[i] <= pos2[i] or pos2[i] + size2[i] <= pos1[i]:
            return False
    return True

class BPPEnv:
    def __init__(self, container_size=(100, 100, 100), max_items=20, grid_size=10):
        """
        container_size: 容器的尺寸 (L, W, H)
        max_items: 最大物品数量
        grid_size: 占用网格的尺寸，用于空间状态编码
        """
        self.container_size = container_size
        self.max_items = max_items
        self.grid_size = grid_size
        self.max_steps = 50  # 限制回合步数，防止无限循环

        self.last_placed_pos = None  # 用于记录上次放置使用的位置(可选)
        self.reset()

    def reset(self, items_list=None):
        if items_list is None:
            self.items = generate_items_by_algorithm1(N=self.max_items, container_size=self.container_size)
        else:
            self.items = copy.deepcopy(items_list)

        self.items_left_idx = list(range(len(self.items)))
        self.placed_items = []
        self.num_steps = 0
        self.done = False

        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        self.candidate_positions = [(0.0, 0.0, 0.0)]  # 初始候选位置
        self.last_placed_pos = None  # 重置

        return self._get_obs()

    def _get_obs(self):
        """
        状态包括：未放置物品信息 + 已放置物品信息 + 容器信息 + 候选位置(坐标) + 占用网格
        """
        items_info = []
        for idx in self.items_left_idx:
            item = self.items[idx]
            if not (isinstance(item, tuple) and len(item) == 3):
                print(f"Error: Item at index {idx} has invalid size {item}")
                continue
            l, w, h = item
            vol = l * w * h
            items_info.append([l, w, h, vol, 0])  # flag=0 未放置

        for pit in self.placed_items:
            size = pit['size']
            if not (isinstance(size, tuple) and len(size) == 3):
                print(f"Error: Placed item has invalid size {size}")
                continue
            l, w, h = size
            vol = l * w * h
            items_info.append([l, w, h, vol, 1])  # flag=1 已放置

        cl, cw, ch = self.container_size
        cvol = cl * cw * ch
        items_info.append([cl, cw, ch, cvol, 2])  # flag=2 容器

        desired_length = 21
        current_length = len(items_info)
        if current_length < desired_length:
            padding_length = desired_length - current_length
            padding = [[0.0, 0.0, 0.0, 0.0, -1]] * padding_length
            items_info.extend(padding)
        elif current_length > desired_length:
            print(f"Warning: Sequence length {current_length} exceeds {desired_length}")
            items_info = items_info[:desired_length]

        positions_info = []
        for pos in self.candidate_positions:
            x, y, z = pos
            x_norm = x / cl
            y_norm = y / cw
            z_norm = z / ch
            positions_info.append([x_norm, y_norm, z_norm])

        return items_info, positions_info, self.occupancy_grid.copy()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        position_idx = action['position_idx']
        item_idx = action['item_idx']

        if position_idx < 0 or position_idx >= len(self.candidate_positions):
            # 无效位置
            reward = -0.2
            self.num_steps += 1
            done = (self.num_steps >= self.max_steps)
            if done:
                reward += self._calc_occupancy()
            return self._get_obs(), reward, done, {}

        position = self.candidate_positions[position_idx]

        # “-2” 表示无可行物品
        if item_idx == -2:
            reward = -0.2
            self.num_steps += 1
            done = (self.num_steps >= self.max_steps)
            if done:
                reward += self._calc_occupancy()
            return self._get_obs(), reward, done, {}

        if item_idx < 0 or item_idx >= len(self.items):
            # 非法物品索引
            reward = -0.2
            self.num_steps += 1
            done = (self.num_steps >= self.max_steps)
            if done:
                reward += self._calc_occupancy()
            return self._get_obs(), reward, done, {}

        if item_idx not in self.items_left_idx:
            # 物品已放置过
            reward = -0.2
            self.num_steps += 1
            done = (self.num_steps >= self.max_steps)
            if done:
                reward += self._calc_occupancy()
            return self._get_obs(), reward, done, {}

        # 尝试放置
        can_place, placed_pos, placed_size = self._try_place_item_with_rotation(self.items[item_idx], position)
        if can_place:
            self.placed_items.append({'pos': placed_pos, 'size': placed_size})
            self._update_occupancy(placed_pos, placed_size)
            item_volume = placed_size[0] * placed_size[1] * placed_size[2]
            container_volume = self.container_size[0] * self.container_size[1] * self.container_size[2]
            reward = item_volume / container_volume
            self.items_left_idx.remove(item_idx)

            # 更新候选位置
            self._update_candidate_positions(placed_pos, placed_size)
            # 记录一下刚用过的位置
            self.last_placed_pos = placed_pos

        else:
            # 放置失败
            reward = -0.1

        self.num_steps += 1
        done = False
        if len(self.items_left_idx) == 0:
            done = True
            reward += self._calc_occupancy()
        elif self.num_steps >= self.max_steps:
            done = True
            reward += self._calc_occupancy()

        self.done = done
        return self._get_obs(), reward, done, {}

    # ---- 6种旋转 ----
    def _try_place_item_with_rotation(self, item, position):
        l, w, h = item
        rotations = [
            (l, w, h),
            (l, h, w),
            (w, l, h),
            (w, h, l),
            (h, l, w),
            (h, w, l),
        ]
        for rot_size in rotations:
            if self._can_place(rot_size, position):
                return True, position, rot_size
        return False, None, None

    def _can_place(self, size, position):
        cl, cw, ch = self.container_size
        il, iw, ih = size
        x, y, z = position

        if x < 0 or y < 0 or z < 0:
            return False
        if x + il > cl or y + iw > cw or z + ih > ch:
            return False

        for pit in self.placed_items:
            if check_overlap((x, y, z), (il, iw, ih), pit['pos'], pit['size']):
                return False
        return True

    def _update_occupancy(self, position, size):
        x, y, z = position
        il, iw, ih = size
        grid_x = int(x / self.container_size[0] * self.grid_size)
        grid_y = int(y / self.container_size[1] * self.grid_size)
        grid_z = int(z / self.container_size[2] * self.grid_size)

        grid_l = max(1, int(il / self.container_size[0] * self.grid_size))
        grid_w = max(1, int(iw / self.container_size[1] * self.grid_size))
        grid_h = max(1, int(ih / self.container_size[2] * self.grid_size))

        grid_x_end = min(self.grid_size, grid_x + grid_l)
        grid_y_end = min(self.grid_size, grid_y + grid_w)
        grid_z_end = min(self.grid_size, grid_z + grid_h)

        self.occupancy_grid[grid_x:grid_x_end, grid_y:grid_y_end, grid_z:grid_z_end] = 1.0

    def _calc_occupancy(self):
        cl, cw, ch = self.container_size
        cont_vol = cl * cw * ch
        used = 0
        for p in self.placed_items:
            used += (p['size'][0] * p['size'][1] * p['size'][2])
        return used / cont_vol

    def _update_candidate_positions(self, placed_pos, placed_size):
        """
        基于当前放置物体，在六个方向生成新的候选位置
        然后调用 `_prune_candidate_positions()` 来删除失效位置
        """
        x, y, z = placed_pos
        l, w, h = placed_size

        directions = [
            ( l, 0, 0),
            (-l, 0, 0),
            (0,  w, 0),
            (0, -w, 0),
            (0, 0,  h),
            (0, 0, -h),
        ]
        for dx, dy, dz in directions:
            new_pos = (x + dx, y + dy, z + dz)
            if new_pos not in self.candidate_positions:
                self.candidate_positions.append(new_pos)

        # 放完新位置后，对 candidate_positions 进行筛除
        self._prune_candidate_positions()

    def _prune_candidate_positions(self):
        """
        删除已经失效的候选位置：
        1) 超出容器
        2) 与已放置物体重叠
        3) (可选) 就是刚放置的旧位置
        """
        valid_positions = []
        for pos in self.candidate_positions:
            if not self._is_position_valid(pos):
                continue

            # 如果想把刚用过的放置位置删除，也可以这样： 
            # if self.last_placed_pos is not None and pos == self.last_placed_pos:
            #     continue

            valid_positions.append(pos)

        self.candidate_positions = valid_positions

    def _is_position_valid(self, pos):
        """ 检查 pos 是否在容器内，不与已放置物体重叠 """
        # 这里假设一个最小尺寸(1,1,1)来做简单判断，也可以更精细化
        min_test_size = (1, 1, 1)
        if not self._is_within_container(pos, min_test_size):
            return False

        # 再检查与已放置物体是否重叠(可以假设 min_test_size 来检测)
        for pit in self.placed_items:
            if check_overlap(pos, min_test_size, pit['pos'], pit['size']):
                return False

        return True

    def _is_within_container(self, pos, size):
        x, y, z = pos
        l, w, h = size
        cl, cw, ch = self.container_size

        if x < 0 or y < 0 or z < 0:
            return False
        if x + l > cl or y + w > cw or z + h > ch:
            return False
        return True

    def visualize_placement(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cl, cw, ch = self.container_size
        ax.bar3d(0, 0, 0, cl, cw, ch, color='cyan', alpha=0.1, edgecolor='black')

        for pit in self.placed_items:
            x, y, z = pit['pos']
            l, w, h = pit['size']
            ax.bar3d(x, y, z, l, w, h, color='blue', alpha=0.5, edgecolor='black')

        ax.set_xlim([0, cl])
        ax.set_ylim([0, cw])
        ax.set_zlim([0, ch])
        plt.show()

############################
# 3. 多头注意力 Transformer + PPO
############################

def sequence_to_tensor_position_first(seq_obs, occupancy_grid, device='cpu'):
    seq_arr = np.array(seq_obs, dtype=np.float32)
    seq_arr[:, :3] /= 100.0  # 归一化长宽高
    seq_arr[:, 3] /= 100.0**3
    seq_tensor = torch.tensor(seq_arr, device=device).unsqueeze(0)  # [1,21,5]

    grid_arr = np.expand_dims(occupancy_grid, axis=0)  # [1,grid_size,grid_size,grid_size]
    grid_arr = np.expand_dims(grid_arr, axis=1)        # [1,1,grid_size,grid_size,grid_size]
    grid_tensor = torch.tensor(grid_arr, device=device)

    return seq_tensor, grid_tensor

class TransformerPolicyNet(nn.Module):
    def __init__(self, max_items=20, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, grid_size=20, num_positions=100):
        super().__init__()
        self.feat_dim = 5
        self.d_model = d_model
        self.max_items = max_items
        self.grid_size = grid_size
        self.num_positions = num_positions

        self.input_fc = nn.Linear(self.feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten(),
            nn.Linear((grid_size // 8) ** 3 * 128, d_model),
            nn.ReLU()
        )

        self.merge_fc = nn.Linear(d_model * 2, d_model)

        self.actor_pos = nn.Linear(d_model, self.num_positions)  
        self.actor_item = nn.Linear(d_model, self.max_items)     
        self.critic = nn.Linear(d_model, 1)

    def forward(self, items_seq, positions_seq, grid):
        x = self.input_fc(items_seq)    # [batch,21,d_model]
        x = self.transformer(x)         # [batch,21,d_model]
        x_pool = x.mean(dim=1)          # [batch,d_model]

        grid_feat = self.conv(grid)     # [batch,d_model]
        combined = torch.cat([x_pool, grid_feat], dim=-1)  # [batch,2*d_model]
        combined = self.merge_fc(combined)                  # [batch,d_model]

        pos_logits = self.actor_pos(combined)    # [batch,num_positions]
        item_logits = self.actor_item(combined)  # [batch,max_items]
        state_value = self.critic(combined)      # [batch,1]

        return pos_logits, item_logits, state_value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs_item = []
        self.logprobs_pos = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

class PPOAgent:
    def __init__(self, max_items=20, lr=1e-4, gamma=0.99, k_epochs=4,
                 eps_clip=0.2, device='cpu', grid_size=20, num_positions=100):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.device = device

        self.policy = TransformerPolicyNet(
            max_items=max_items, grid_size=grid_size, 
            num_positions=num_positions
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)

        self.old_policy = TransformerPolicyNet(
            max_items=max_items, grid_size=grid_size, 
            num_positions=num_positions
        ).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()

        self.buffer = RolloutBuffer()

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def select_action(self, items_seq, occupancy_grid, env):
        self.policy.eval()
        with torch.no_grad():
            seq_tensor, grid_tensor = sequence_to_tensor_position_first(items_seq, occupancy_grid, device=self.device)
            pos_logits, item_logits, state_value = self.old_policy(seq_tensor, None, grid_tensor)

        pos_probs = torch.softmax(pos_logits, dim=-1)  
        dist_pos = Categorical(pos_probs)
        position_idx = dist_pos.sample()
        logprob_pos = dist_pos.log_prob(position_idx).squeeze()

        selected_position = env.candidate_positions[position_idx.item()]

        # 找可行物品
        feasible_items = []
        for idx in env.items_left_idx:
            can_place, _, _ = env._try_place_item_with_rotation(env.items[idx], selected_position)
            if can_place:
                feasible_items.append(idx)

        # 如果没有可行物品，则返回 item_idx=-2
        if not feasible_items:
            action = {
                'position_idx': position_idx.item(),
                'item_idx': -2
            }
            return action, torch.tensor(-0.1, device=self.device), logprob_pos, state_value.item()

        # 否则，对不可行物品 logits设为很小
        item_logits_masked = item_logits.clone()
        mask = torch.zeros(self.policy.max_items, dtype=torch.bool, device=self.device)
        for idx in feasible_items:
            mask[idx] = True
        mask = mask.view(1, -1)
        item_logits_masked[~mask] = -1e10

        item_probs = torch.softmax(item_logits_masked, dim=-1)
        dist_item = Categorical(item_probs)
        item_idx = dist_item.sample()
        logprob_item = dist_item.log_prob(item_idx).squeeze()

        selected_item_idx = item_idx.item()
        action = {
            'position_idx': position_idx.item(),
            'item_idx': selected_item_idx
        }
        return action, logprob_item, logprob_pos, state_value.item()

    def update(self, env):
        if len(self.buffer.rewards) == 0:
            print("Buffer is empty, no update.")
            return

        states = self.buffer.states
        actions = self.buffer.actions

        logprobs_item_list = [t.squeeze() for t in self.buffer.logprobs_item if t.numel() > 0]
        logprobs_pos_list = [t.squeeze() for t in self.buffer.logprobs_pos if t.numel() > 0]

        if len(logprobs_item_list) == 0 or len(logprobs_pos_list) == 0:
            print("logprobs_item_list or logprobs_pos_list is empty, skip update.")
            return

        logprobs_item = torch.stack(logprobs_item_list).to(self.device)
        logprobs_pos = torch.stack(logprobs_pos_list).to(self.device)

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float, device=self.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float, device=self.device)

        # 计算回报
        returns = []
        discounted_sum = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d == 1.0:
                discounted_sum = 0
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = returns - old_values

        items_seqs = [s[0] for s in states]
        occupancy_grids = [s[1] for s in states]

        try:
            seq_tensors, grid_tensors = zip(*[
                sequence_to_tensor_position_first(obs, grid, device=self.device)
                for obs, grid in zip(items_seqs, occupancy_grids)
            ])
            seq_tensors = torch.cat(seq_tensors, dim=0)
            grid_tensors = torch.cat(grid_tensors, dim=0)
        except Exception as e:
            print(f"Error in tensor concatenation: {e}")
            return

        for epoch in range(self.k_epochs):
            pos_logits, item_logits, state_values = self.policy(seq_tensors, None, grid_tensors)
            position_indices = torch.tensor([a['position_idx'] for a in actions], device=self.device)
            item_indices = torch.tensor([a['item_idx'] for a in actions], device=self.device)

            # 只保留合法的
            mask = (item_indices >= 0) & (position_indices >= 0)
            if not mask.any():
                print("No valid actions to update.")
                return

            position_indices = position_indices[mask]
            item_indices = item_indices[mask]
            logprobs_item_masked = logprobs_item[mask]
            logprobs_pos_masked = logprobs_pos[mask]
            returns_epoch = returns[mask]
            advantages_epoch = advantages[mask]
            state_values = state_values.squeeze()[mask]

            pos_logits = pos_logits[mask]
            item_logits = item_logits[mask]

            dist_pos = Categorical(logits=pos_logits)
            logprobs_pos_new = dist_pos.log_prob(position_indices)

            dist_item = Categorical(logits=item_logits)
            logprobs_item_new = dist_item.log_prob(item_indices)

            ratios_pos = torch.exp(logprobs_pos_new - logprobs_pos_masked)
            ratios_item = torch.exp(logprobs_item_new - logprobs_item_masked)
            ratios = ratios_pos * ratios_item

            surr1 = ratios * advantages_epoch
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_epoch
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(state_values, returns_epoch)

            entropy_pos = dist_pos.entropy()
            entropy_item = dist_item.entropy()
            entropy = entropy_pos + entropy_item
            entropy_loss = -0.01 * entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.scheduler.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def store_transition(self, state, action, logprob_item, logprob_pos, val, reward, done):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        # squeeze成标量
        self.buffer.logprobs_item.append(
            logprob_item.unsqueeze(0) if logprob_item.dim() == 0 else logprob_item
        )
        self.buffer.logprobs_pos.append(
            logprob_pos.unsqueeze(0) if logprob_pos.dim() == 0 else logprob_pos
        )
        self.buffer.values.append(val)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(1.0 if done else 0.0)

############################
# 4. 训练 & 测试
############################

def train_ppo(env, agent, train_data, num_episodes=5000, update_timestep=1000, writer=None):
    timestep = 0
    for ep in range(num_episodes):
        items_list = random.choice(train_data)
        seq_obs, _, occupancy_grid = env.reset(items_list=items_list)
        done = False
        episode_reward = 0.0
        num_placed = 0

        while not done:
            action, logprob_item, logprob_pos, val = agent.select_action(seq_obs, occupancy_grid, env)
            seq_next, reward, done, _ = env.step(action)

            seq_next_obs, _, next_occupancy_grid = seq_next

            if reward > 0:  # 如果放置成功
                num_placed += 1

            agent.store_transition(
                (seq_obs, occupancy_grid),
                action,
                logprob_item,
                logprob_pos,
                val,
                reward,
                done
            )

            seq_obs = seq_next_obs
            occupancy_grid = next_occupancy_grid
            episode_reward += reward
            timestep += 1

            if timestep % update_timestep == 0:
                agent.update(env)

        # 每50回合评估一次
        if (ep + 1) % 50 == 0:
            avg_occ = evaluate(env, agent, num_eps=20)
            if writer:
                writer.add_scalar('Reward/Episode', episode_reward, ep + 1)
                writer.add_scalar('Occupancy/Avg', avg_occ, ep + 1)
                writer.add_scalar('Items/Placed', num_placed, ep + 1)
            print(f"Episode {ep+1}/{num_episodes}, Reward={episode_reward:.2f}, "
                  f"Items Placed={num_placed}, Avg Occupancy={avg_occ*100:.2f}%")

def evaluate(env, agent, num_eps=10):
    total_occ = 0.0
    for _ in range(num_eps):
        seq_obs, _, occupancy_grid = env.reset()
        done = False
        while not done:
            action, _, _, _ = agent.select_action(seq_obs, occupancy_grid, env)
            seq_next, reward, done, _ = env.step(action)
            seq_obs, _, occupancy_grid = seq_next
        occ = env._calc_occupancy()
        total_occ += occ
    return total_occ / num_eps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    writer = SummaryWriter()

    train_data = create_dataset(num_samples=500, max_items=10, container_size=(100, 100, 100))
    test_data  = create_dataset(num_samples=50,  max_items=10, container_size=(100, 100, 100))

    env = BPPEnv(container_size=(100, 100, 100), max_items=10, grid_size=20)

    agent = PPOAgent(
        max_items=10, 
        lr=1e-4,        
        gamma=0.99,     
        eps_clip=0.2,   
        device=device, 
        grid_size=20,
        num_positions=len(env.candidate_positions)
    )

    print("开始训练(PPO)...")
    train_ppo(env, agent, train_data, num_episodes=5000, update_timestep=1000, writer=writer)

    avg_occ = evaluate(env, agent, num_eps=20)
    print(f"测试: 平均空间利用率 = {avg_occ*100:.2f}%")

    writer.close()
    env.visualize_placement()

if __name__ == "__main__":
    main()
