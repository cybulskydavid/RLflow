import os
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
                 max_env_steps: int,
                 warmup_steps: int,
                 steps_per_save: int,
                 save_path: str):
        super().__init__(runner, algorithm, logger, save_path, steps_per_save)
        self.max_env_steps = max_env_steps
        self.warmup_steps = warmup_steps

    def train(self):
        print(f"Start treningu Off-Policy (Max Steps: {self.max_env_steps}, Warmup: {self.warmup_steps})")
        
        while self.runner.global_step < self.max_env_steps:
    
            self.runner.run()
            
            if self.runner.global_step > self.warmup_steps:
                stats = self.algorithm.update(self.runner.agent, self.runner.buffer)
                
                self.logger.log_metrics(stats, self.runner.global_step)

            if self.runner.global_step % self.steps_per_save == 0:
                os.makedirs(self.save_path, exist_ok=True)
                state_dict = self.algorithm.get_state_dict(self.runner.agent)
                torch.save(state_dict, f"{self.save_path}/checkpoint_{self.runner.global_step}.pth")
                print(f"Saved checkpoint at step {self.runner.global_step}")