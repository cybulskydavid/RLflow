from typing import Callable, Dict
from agents.base import BaseAgent
from agents.continuous import ContinuousAgent
from configs.agent import BaseAgentConfig, ContinuousAgentConfig
from networks.factory import make_architecture


type AgentBuilderFn = Callable[[int, int, BaseAgentConfig], BaseAgent]


def build_continuous_agent(obs_dim: int, action_dim: int, cfg: ContinuousAgentConfig) -> ContinuousAgent:
    architecture = make_architecture(obs_dim, action_dim, cfg.architecture)

    if architecture is None:
        return None
    
    return ContinuousAgent(architecture)


AGENTS_BUILDERS:Dict[str, AgentBuilderFn] = {
    "continuous_agent": build_continuous_agent
}


def make_agent(obs_dim: int, action_dim: int, cfg: BaseAgentConfig) -> BaseAgent:
    agent_builder = AGENTS_BUILDERS.get(cfg.type)

    if agent_builder is None:
        return None

    return agent_builder(obs_dim, action_dim, cfg)