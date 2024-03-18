import random
import gymnasium as gym
import numpy as np
from smart_bed_env import SmartBedEnv
import os
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

### experience replay
# 构造批次化训练数据
# 让整个训练过程更稳定

# s_t, a_t => s_{t+1}
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# tr = Transition(1, 2, 3, 4)
# tr.state # 1

# trs = [Transition(1, 2, 3, 4), Transition(5, 6, 7, 8)]
# trs = Transition(*zip(*trs))
# trs  # Transition(state=(1, 5), action=(2, 6), next_state=(3, 7), reward=(4, 8))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(*args)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, num_actions_per_dim, num_dims):
        super(DQN, self).__init__()
        self.num_actions_per_dim = num_actions_per_dim
        self.num_dims = num_dims
        output_size = num_actions_per_dim * num_dims  # 每个维度的动作数 * 动作维度数

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 将输出重塑为每个动作维度的概率分布
        x = x.view(-1, self.num_dims, self.num_actions_per_dim)
        return x


class Agent:
    def __init__(self, env):
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.nvec.prod()  # Correct way to get total number of actions for MultiDiscrete
        self.gamma = 0.99
        self.batch_size = 32
        self.capacity = 10000
        self.memory = ReplayMemory(self.capacity)
        self.model = DQN(self.n_states, env.action_space.nvec[0], len(env.action_space.nvec))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)  # 从经验回放缓冲区（Experience Replay Buffer）中随机抽样一批经验样本以供训练
        batch = Transition(*zip(*batch))  # 将list of transition 转化成一个transition, column: len(tuple) == batch_size

        state_batch = torch.cat(batch.state)  # s_t为4D，即s_t.shape == batch_size * 4 (pos, cart_v, angle, pole_v)
        action_batch = torch.cat(batch.action)  # a_t.shape == batch_size * 1
        reward_batch = torch.cat(batch.reward)  # r_{t+1}.shape == batch_size * 1
        # next_state_batch = torch.cat(batch.next_state)
        # non_final_next_state_batch = torch.cat(
        #     [s for s in batch.next_state if s is not None])  # 将符合条件的s拼接成一个张量，可能小于batch_size，即有些遇到结束状态

        non_final_next_state_batch = [s for s in batch.next_state if s is not None]
        if len(non_final_next_state_batch) > 0:
            non_final_next_state_batch = torch.cat(non_final_next_state_batch)
        else:
            non_final_next_state_batch = None

        # 构造model的input和output
        # input: s_t
        # pred: Q(s_t, a_t)
        # true: R_{t+1} + gamma * max(Q(s_{t+1}, a))
        # purpose: pred approach true

        # 开启eval模式
        self.model.eval()

        # pred
        # 维度: batch_size * 1
        state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)

        # true
        # tuple(map(lambda s: s is not None, batch.next_state)) 为batch_size长度的0/1
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))  # 用于标识批量中哪些样本具有非空的下一个状态next_state (1)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_state_batch).max(dim=1)[0].detach()  # 计算非终止状态下的下一状态值，max(dim=1)[0]为状态值的最大估计

        # 维度: (batch_size, )
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # 开启train模式
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))  # 增加一个维度对齐

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def update_q_function(self):
        self._replay()

    def choose_action(self, state, episode):
        eps = 0.5 * 1 / (1 + episode)
        if random.random() < eps:
            # explore
            action = torch.tensor([random.randrange(num) for num in env.action_space.nvec], dtype=torch.int64)
        else:
            # exploit
            self.model.eval()
            with torch.no_grad():
                logits = self.model(state)  # [1, num_dims, num_actions_per_dim]
                probabilities = F.softmax(logits, dim=-1)  # 应用 softmax 得到概率分布
                action = probabilities.max(2)[1].view(-1)  # 选择概率最高的动作
        return action.numpy()


# interaction between agent and environment
env = gym.make('SmartBedEnv-v0')
n_states = env.observation_space.shape[0]  # 16
n_actions = env.action_space.shape or env.action_space.n  # 6

agent = Agent(env)

max_episodes = 50000
max_steps = 200

learning_finish_flag = False
frames = []

for episode in range(max_episodes):
    print("--------------------------")
    print("Episode: ", episode)
    state, _ = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0)

    for step in range(max_steps):
        print()
        print("Step: ", step)
        action = agent.choose_action(state, episode)
        next_state, reward, done, _, _ = env.step(action)  # Adjust for MultiDiscrete
        reward = torch.tensor([reward], dtype=torch.float)

        if done:
            next_state = None
        else:
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        agent.memory.push(state, action, next_state, reward)
        agent.update_q_function()
        state = next_state

        if done:
            print(f'Episode: {episode}, Steps: {step}')
            break
