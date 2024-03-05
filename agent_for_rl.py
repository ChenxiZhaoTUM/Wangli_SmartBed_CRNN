import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from smart_bed_env import SmartBedEnv
from smart_bed_env_test import SmartBedEnvTest


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


def train():
    envs = gym.make('SmartBedEnv-v0')
    test_envs = gym.make('SmartBedEnvTest-v0')

    num_envs = 1
    # envs = SubprocVectorEnv([make_env for _ in range(num_envs)])
    # test_envs = SubprocVectorEnv([make_env for _ in range(num_envs)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    observation_shape = envs.observation_space.shape or envs.observation_space.n
    action_shape = envs.action_space.shape or envs.action_space.n

    net = Net(observation_shape, action_shape)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    policy = DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

    # lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / 300)

    train_collector = Collector(policy, envs, VectorReplayBuffer(4000, num_envs), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    log_path = "log"
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=100, step_per_epoch=1000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=None,
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')


if __name__ == "__main__":
    train()
