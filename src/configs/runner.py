from dataclasses import MISSING, dataclass


@dataclass
class RunnerConfig:
    type: str = MISSING


@dataclass
class OnPolicyRunnerConfig(RunnerConfig):
    type: str = "on_policy"