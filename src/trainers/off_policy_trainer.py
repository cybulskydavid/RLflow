import torch
from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner
from trainers.base_trainer import BaseTrainer
from utils.tensor_board_logger import TensorBoardLogger

class OffPolicyTrainer(BaseTrainer):
    def __init__(self, 
                 runner: BaseRunner, 
                 algorithm: BaseAlgorithm,
                 logger: TensorBoardLogger,
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
                
                self.logger.log_metrics(stats, self.runner.global_step)