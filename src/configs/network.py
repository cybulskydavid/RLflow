from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class LayerConfig:
    type: str


@dataclass
class LinearConfig(LayerConfig):
    type: str = "linear"
    out_features: int = 256
    bias: bool = True


@dataclass
class Conv2dConfig(LayerConfig):
    type: str = "conv2d"
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0


@dataclass
class MaxPool2dConfig(LayerConfig):
    type: str = "maxpool2d"
    kernel_size: int = 2
    stride: int = 2


@dataclass
class FlattenConfig(LayerConfig):
    type: str = "flatten"


@dataclass
class ReLUConfig(LayerConfig):
    type: str = "ReLU"


@dataclass
class ExtractorConfig:
    features_dim: int
    layer_definitions: List[Any]


@dataclass
class MLPExtractorConfig(ExtractorConfig):
    features_dim: int = 256
    layer_definitions: List[Any] = field(default_factory=list)


@dataclass
class CNNExtractorConfig(ExtractorConfig):
    features_dim: int = 512
    layer_definitions: List[Any] = field(default_factory=list)