from dataclasses import MISSING, dataclass


@dataclass
class RunnerConfig:
    type: str = MISSING


@dataclass(kw_only=True)
class OnPolicyRunnerConfig(RunnerConfig):
    type: str = "on_policy"


@dataclass(kw_only=True)
class OffPolicyRunnerConfig(RunnerConfig):
    type: str = "off_policy"
    steps_per_run: int = 1      
    start_steps: int = 10_000   
    seed: int = 42