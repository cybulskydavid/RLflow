from typing import Callable, Dict
from configs.network import BaseArchitectureConfig, SeparatedArchitectureConfig, VectorHeadConfig
from networks.architectures import SeparatedArchitecture
from networks.extractors import Extractor
from networks.heads import IndependentStdHead, ScalarHead, VectorHead

type BuildVectorHeadFn = Callable[[int, int], VectorHead]
type BuildArchitectureFn = Callable[[int, int, BaseArchitectureConfig], BaseArchitectureConfig]


def build_vector_head(feature_dim: int, action_dim: int) -> VectorHead:
    return VectorHead(feature_dim, action_dim)


def build_independent_std_head(feature_dim: int, action_dim: int):
    return IndependentStdHead(feature_dim, action_dim, 1)


VECTOR_HEAD_BUILDERS: Dict[str, BuildVectorHeadFn] = {
    "vector_head": build_vector_head,
    "independent_std_head": build_independent_std_head
}


def build_shared_architecture():
    pass


def build_separated_architecture(obs_dim: int, action_dim: int, cfg: SeparatedArchitectureConfig):
    build_actor_head = VECTOR_HEAD_BUILDERS.get(cfg.actor_head.type)

    print(build_actor_head)

    if build_actor_head is None:
        return None

    actor_extractor = Extractor(obs_dim[0], cfg.actor_extractor.layer_definitions)
    critic_extractor = Extractor(obs_dim[0], cfg.critic_extractor.layer_definitions)
    actor_head = build_actor_head(actor_extractor.feature_dim, action_dim[0])
    critic_head = ScalarHead(critic_extractor.feature_dim)

    return SeparatedArchitecture(actor_extractor, critic_extractor, actor_head, critic_head)



ARCHITECTURE_BUILDERS: Dict[str, BuildArchitectureFn] = {
    "shared_architecture": build_shared_architecture,
    "separated_architecture": build_separated_architecture
}


def make_architecture(obs_dim: int, action_dim: int, cfg: BaseArchitectureConfig):
    build_architecture = ARCHITECTURE_BUILDERS.get(cfg.type)

    print(build_architecture)

    if build_architecture is None:
        return None
    
    return build_architecture(obs_dim, action_dim, cfg)