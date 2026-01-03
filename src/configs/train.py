from dataclasses import MISSING, dataclass

from configs.buffer import BufferConfig
from configs.env import EnvConfig


@dataclass
class TrainConfig:
    env: EnvConfig = MISSING
    buffer: BufferConfig = MISSING
    seed: int = 0

