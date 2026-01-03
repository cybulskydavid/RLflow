import torch
import random

import numpy as np
import hydra
from hydra.utils import instantiate

from configs import register_configs
from configs.train import TrainConfig
from envs.factory import make_env
from envs.gym_env import GymEnv

import os


def set_glogal_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: TrainConfig):
    cfg= instantiate(cfg)

    print(cfg)

    set_glogal_seed(cfg.seed)
    env = make_env(cfg.env)

    print(cfg)

    env.reset(cfg.seed)

    while True:
        action = env.env.action_space.sample()
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated:
            env.reset()


if __name__ == "__main__":
    register_configs()
    main()