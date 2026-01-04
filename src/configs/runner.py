from dataclasses import MISSING, dataclass


@dataclass
class RunnerConfig:
    type: str = MISSING


@dataclass(kw_only=True)
class OnPolicyRunnerConfig(RunnerConfig):
    type: str = "on_policy"