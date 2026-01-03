from dataclasses import MISSING, dataclass


@dataclass
class EnvConfig:
    id: str = MISSING
    type: str = MISSING
    

@dataclass
class GymEnvConfig(EnvConfig):
    type: str = "gym"
    render_mode: str | None = None
    max_episode_steps: int = 1600