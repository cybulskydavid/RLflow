from dataclasses import MISSING, dataclass

from configs.buffer import BufferConfig
from configs.env import EnvConfig
from configs.runner import RunnerConfig


@dataclass
class TrainConfig:
    env: EnvConfig = MISSING
    buffer: BufferConfig = MISSING
    runner: RunnerConfig = MISSING
    seed: int = 0

