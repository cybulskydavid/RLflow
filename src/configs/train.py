from dataclasses import MISSING, dataclass

from configs.env import EnvConfig


@dataclass
class TrainConfig:
    env: EnvConfig = MISSING
    seed: int = 0

