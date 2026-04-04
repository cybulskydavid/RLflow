from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from torch import Tensor
from envs.base_env import BaseEnv, EnvSpec


class GymEnv(BaseEnv):
    def __init__(self, id: str, render_mode: str = None) -> None:
        self.env = gym.make(id, render_mode=render_mode)

        self._spec = EnvSpec(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            action_type="discrete" if isinstance(self.env.action_space, gym.spaces.Discrete) else "continuous",
            action_min=self.env.action_space.low,
            action_max=self.env.action_space.high
        )

    @property
    def spec(self) -> EnvSpec:
        return self._spec


    def reset(self, seed: int = None) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset(seed=np.random.randint(0, 10000))


    def step(self, 
             action: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if isinstance(action, Tensor):
            action = action.numpy()
        return self.env.step(action)


    def close(self) -> None:
        return self.env.close()