import os
import torch
from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner
from utils.tensor_board_logger import TensorBoardLogger


class BaseTrainer:
    def __init__(self, 
                 runner: BaseRunner, 
                 algorithm: BaseAlgorithm, 
                 logger: TensorBoardLogger,
                 save_path: str,
                 steps_per_save: int):
        self.runner = runner
        self.algorithm = algorithm
        self.logger = logger
        self.save_path = save_path
        self.steps_per_save = steps_per_save

    def train(self):
        for step in range(100_000):
            self.runner.run()
            stats = self.algorithm.update(self.runner.agent, self.runner.buffer)
            self.logger.log_metrics(stats, step)

            if step % self.steps_per_save == 0:
                os.makedirs(self.save_path, exist_ok=True)
                state_dict = self.algorithm.get_state_dict(self.runner.agent)
                torch.save(state_dict, f"{self.save_path}/checkpoint_{step}.pth")
                print(f"Saved checkpoint at step {step}")