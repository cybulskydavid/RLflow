import torch
from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner
from trainers.base_trainer import BaseTrainer

class OffPolicyTrainer(BaseTrainer):
    def __init__(self, 
                 runner: BaseRunner, 
                 algorithm: BaseAlgorithm,
                 logger: any,
                 max_env_steps: int = 1_000_000,
                 warmup_steps: int = 10_000,
                 log_freq: int = 1000):
        
        super().__init__(runner, algorithm, logger)
        self.max_env_steps = max_env_steps
        self.warmup_steps = warmup_steps
        self.log_freq = log_freq

    def train(self):
        print(f"Start treningu Off-Policy (Max Steps: {self.max_env_steps}, Warmup: {self.warmup_steps})")
        
        while self.runner.global_step < self.max_env_steps:
    
            self.runner.run()
            
            if self.runner.global_step > self.warmup_steps:
                stats = self.algorithm.update(self.runner.agent, self.runner.buffer)
                
                if self.runner.global_step % self.log_freq == 0:
                    self._log_stats(stats, self.runner.global_step)

    def _log_stats(self, stats: dict, step: int):
        log_str = f"Step: {step} | "
        for k, v in stats.items():
            log_str += f"{k}: {v:.4f} | "
        print(log_str)