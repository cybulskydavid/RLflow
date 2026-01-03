from typing import Callable, Dict
from agents.base import BaseAgent
from buffers.base_buffer import BaseBuffer
from configs.runner import OnPolicyRunnerConfig, RunnerConfig
from envs.base_env import BaseEnv
from runners.base_runner import BaseRunner
from runners.on_policy_runner import OnPolicyRunner


type BuildRunnerFn = Callable[[BaseEnv, BaseAgent, BaseBuffer, RunnerConfig], BaseRunner]


def build_on_policy_runner(env: BaseEnv, agent: BaseAgent, buffer: BaseBuffer, cfg: OnPolicyRunnerConfig) -> OnPolicyRunner:
    return OnPolicyRunner(env, agent, buffer, cfg)


RUNNER_BUILDERS: Dict[str, BuildRunnerFn] = {
    "on_policy": build_on_policy_runner
}


def make_runner(env: BaseEnv, agent: BaseAgent, buffer: BaseBuffer, cfg: RunnerConfig) -> BaseRunner:
    runner_builder = RUNNER_BUILDERS.get(cfg.type)
    
    if runner_builder is None:
        return None
    
    return runner_builder(env, agent, buffer, cfg)