from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from envs.base_env import BaseEnv, EnvSpec
from configs.env import GymEnvConfig


class GymEnv(BaseEnv):
    def __init__(self, cfg: GymEnvConfig):
        self.env = gym.make(cfg.id, 
                            render_mode=cfg.render_mode)
        
        self._spec = EnvSpec(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            action_type="discrete" if isinstance(self.env.action_space, gym.spaces.Discrete) else "continuous",
        )

    @property
    def spec(self) -> EnvSpec:
        return self._spec


    def reset(self, seed: int = None) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset(seed=seed)


    def step(self, 
             action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        return self.env.step(action)


    def close(self) -> None:
        return self.env.close()