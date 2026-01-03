from typing import Callable, Dict, Tuple
from buffers.base_buffer import BaseBuffer
from buffers.rollout_buffer import RolloutBuffer
from configs.buffer import BufferConfig, RolloutBufferConfig

type BuildBufferFn = Callable[[Tuple[int, ...], Tuple[int, ...], BufferConfig], BaseBuffer]


def build_rollout_buffer(obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...], cfg: RolloutBufferConfig) -> RolloutBuffer:
    return RolloutBuffer(cfg.buffer_size, obs_shape, act_shape)


BUFFER_BUILDERS: Dict[str, BuildBufferFn] = {
    "rollout": build_rollout_buffer
}


def make_buffer(obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...], cfg: BufferConfig) -> BaseBuffer:
    buffer_builder = BUFFER_BUILDERS.get(cfg.type)
    
    if buffer_builder is None:
        return None
    
    return buffer_builder(obs_shape, act_shape, cfg)