from typing import Callable, Dict

import hydra
from configs.env import EnvConfig
from envs.base_env import BaseEnv
from envs.gym_env import GymEnv

type BuildEnvFn = Callable[[EnvConfig], BaseEnv]


def build_gym_env(cfg:EnvConfig) -> GymEnv:
    return GymEnv(cfg)


env_builders: Dict[str, BuildEnvFn] = {
    "gym": build_gym_env
}


def make_env(cfg: EnvConfig) -> BaseEnv:
    print(cfg)  

    env_builder = env_builders.get(cfg.type)
    
    if env_builder is None:
        return None
    return env_builder(cfg)