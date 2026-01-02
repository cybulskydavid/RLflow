from typing import Callable, Dict, Tuple
from configs import network
from torch import nn

type BuildLayerFn = Callable[[network.LayerConfig, int], Tuple[nn.Module, int]]

def build_linear(cfg: network.LinearConfig, input_dim: int) -> Tuple[nn.Linear, int]:
    layer = nn.Linear(input_dim, cfg.out_features, cfg.bias)
    return layer, cfg.out_features


def build_conv2d(cfg: network.Conv2dConfig, in_channels: int) -> Tuple[nn.Conv2d, int]:
    layer = nn.Conv2d(in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding)
    return layer, cfg.out_channels


def build_maxpool2d(cfg: network.MaxPool2dConfig, in_channels: int) -> Tuple[nn.MaxPool2d, int]:
    layer = nn.MaxPool2d(cfg.kernel_size, cfg.stride)
    return layer, in_channels


def build_flatten(cfg: network.FlattenConfig, input_dim: int) -> Tuple[nn.Flatten, int]:
    return nn.Flatten(), input_dim


def build_relu(cfg: network.ReLUConfig, input_dim: int) -> Tuple[nn.ReLU, int]:
    return nn.ReLU(), input_dim


LAYER_BUILDERS: Dict[str, BuildLayerFn] = {
    "linear": build_linear,
    "conv2d": build_conv2d,
    "maxpool2d": build_maxpool2d,
    "flatten": build_flatten,
    "ReLU": build_relu
}


def build_layer(cfg: network.LayerConfig, input_dim: int) -> Tuple[nn.Module, int]:
    layer_builder = LAYER_BUILDERS.get(cfg.type)
    return layer_builder(cfg, input_dim)