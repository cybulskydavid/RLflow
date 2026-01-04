from omegaconf import DictConfig, OmegaConf
import torch
import random

import numpy as np
import hydra
from hydra.utils import instantiate

from agents.base import BaseAgent
from agents.continuous import ContinuousAgent
from agents.factory import make_agent
from buffers.factory import make_buffer
from configs import register_configs
from configs.train import TrainConfig
from envs.factory import make_env
from envs.gym_env import GymEnv

import os

from runners.factory import make_runner


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):
    
    print(cfg)

    config: TrainConfig = hydra.utils.instantiate(cfg)

    print(config)

    set_global_seed(config.seed)
    env = make_env(config.env)
    buffer = make_buffer(env.spec.obs_shape, env.spec.action_shape, config.buffer)
    agent = make_agent(env.spec.obs_shape, env.spec.action_shape, config.agent)
    runner = make_runner(env, agent, buffer, config.runner)

    runner.run()

if __name__ == "__main__":
    register_configs()
    main()