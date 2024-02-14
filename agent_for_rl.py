import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from smart_bed_env import SmartBedEnv


def make_env():
    return SmartBedEnv()


def train():
    num_envs = 1
    envs = SubprocVectorEnv([make_env for _ in range(num_envs)])
    test_envs = SubprocVectorEnv([make_env for _ in range(num_envs)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    observation_shape = envs.observation_space.shape or envs.observation_space.n
    action_shape = envs.action_space.shape or envs.action_space.n

    net = Net(state_shape=observation_shape, hidden_sizes=[128, 128], activation=nn.Tanh, device=device)
    actor = ActorProb(net, action_shape, max_action=envs.action_space.high[0], device=device).to(device)
    net_c = Net(state_shape=observation_shape, hidden_sizes=[128, 128], activation=nn.Tanh, device=device)
    critic = Critic(net_c, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, 0.5)

    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=1e-3)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / 300)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=0.99,
        eps_clip=0.2,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=False,
        vf_coef=0.25,
        ent_coef=0.0,
        max_grad_norm=None,
        gae_lambda=0.95,
        reward_normalization=False,
        max_batchsize=1024,
        action_scaling=True,
        action_bound_method='tanh',
        action_space=envs.action_space,
        lr_scheduler=lr_scheduler,
        deterministic_eval=True
    )

    train_collector = Collector(policy, envs, VectorReplayBuffer(4000, num_envs))
    test_collector = Collector(policy, test_envs)

    log_path = "log"
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    result = onpolicy_trainer(policy, train_collector, test_collector,
                              max_epoch=500, step_per_epoch=1000, episode_per_test=10, batch_size=64,
                              stop_fn=None, save_checkpoint_fn=None, logger=logger)

    print(f"Finished training. Final reward: {result['best_reward']:.2f}")


if __name__ == "__main__":
    train()
