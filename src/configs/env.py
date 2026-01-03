from dataclasses import dataclass


@dataclass
class EnvConfig:
    id: str = "BipedalWalker-v3"
    type: str = ""
    

@dataclass
class GymEnvConfig(EnvConfig):
    type: str = "gym"
    render_mode: str | None = None
    max_episode_steps: int = 1600