from dataclasses import MISSING, dataclass


@dataclass
class BufferConfig:
    type: str = MISSING
    buffer_size: int = 2048


@dataclass
class RolloutBufferConfig(BufferConfig):
    type: str = "rollout"