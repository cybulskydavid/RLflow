from omegaconf import DictConfig
import torch
import random

import numpy as np
import hydra
from hydra.utils import instantiate
from agents.factory import make_agent
from algorithms.ppo import PPO
from buffers.factory import make_buffer
from configs import register_configs
from configs.train import TrainConfig
from envs.factory import make_env
from envs.gym_env import GymEnv

import os

from runners.factory import make_runner
from trainers.base_trainer import BaseTrainer


lr_actor = 0.00003           # Learning rate
lr_critic = 0.0001

def set_global_seed(seed: int):
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):

    config: TrainConfig = hydra.utils.instantiate(cfg)

    set_global_seed(config.seed)
    env = make_env(config.env)
    buffer = make_buffer(env.spec.obs_shape, env.spec.action_shape, config.buffer)
    agent = make_agent(env.spec.obs_shape, env.spec.action_shape, config.agent)

    # print(agent.architecture)
    # raise Exception("Debug stop")

    runner = make_runner(env, agent, buffer, config.runner)
    optimizer = torch.optim.Adam([
        {'params': agent.architecture.actor_extractor.parameters(), 'lr': lr_actor},
        {'params': agent.architecture.actor_head.parameters(), 'lr': lr_actor},
        {'params': agent.architecture.critic_extractor.parameters(), 'lr': lr_critic},
        {'params': agent.architecture.critic_head.parameters(), 'lr': lr_critic}
    ])
    algorithm = PPO(optimizer)
    trainer = BaseTrainer(runner, algorithm)
    
    trainer.train()

if __name__ == "__main__":
    register_configs()
    main()