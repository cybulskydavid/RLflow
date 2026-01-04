from dataclasses import MISSING, dataclass, field
from typing import Any, List


@dataclass
class LayerConfig:
    type: str


@dataclass(kw_only=True)
class LinearConfig(LayerConfig):
    type: str = "linear"
    out_features: int = 256
    bias: bool = True


@dataclass(kw_only=True)
class Conv2dConfig(LayerConfig):
    type: str = "conv2d"
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0


@dataclass(kw_only=True)
class MaxPool2dConfig(LayerConfig):
    type: str = "maxpool2d"
    kernel_size: int = 2
    stride: int = 2


@dataclass(kw_only=True)
class FlattenConfig(LayerConfig):
    type: str = "flatten"


@dataclass(kw_only=True)
class ReLUConfig(LayerConfig):
    type: str = "ReLU"


@dataclass
class ExtractorConfig:
    layer_definitions: List[Any] = field(default_factory=list)


@dataclass
class VectorHeadConfig:
    type: str = "vector_head"


dataclass
class ScalarHeadConfig:
    type: str = "scalar_head"


@dataclass(kw_only=True)
class IndependentStdHeadConfig(VectorHeadConfig):
    type: str = "independent_std_head"
    initial_log_std: float = 0.0


@dataclass(kw_only=True)
class StateDependentGaussianHeadConfig(VectorHeadConfig):
    type: str = "state_dependent_gaussian_head"
    log_std_min: float = -2
    log_std_max: float = 2


@dataclass
class BaseArchitectureConfig:
    type: str


@dataclass(kw_only=True)
class SharedArchitectureConfig(BaseArchitectureConfig):
    type: str = "shared_architecture"
    shared_extractor: ExtractorConfig = MISSING
    actor_head: VectorHeadConfig = MISSING
    critic_head: ScalarHeadConfig = MISSING


@dataclass(kw_only=True)
class SeparatedArchitectureConfig(BaseArchitectureConfig):
    type: str = "separated_architecture"
    actor_extractor: ExtractorConfig = MISSING
    critic_extractor: ExtractorConfig = MISSING
    actor_head: VectorHeadConfig = MISSING
    critic_head: ScalarHeadConfig = MISSING