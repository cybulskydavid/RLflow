from dataclasses import MISSING, dataclass

from configs.network import BaseArchitectureConfig


@dataclass
class BaseAgentConfig:
    type: str = MISSING
    architecture: BaseArchitectureConfig = MISSING


@dataclass(kw_only=True)
class ContinuousAgentConfig(BaseAgentConfig):
    type: str = "continuous_agent"