import torch
from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner
from utils.tensor_board_logger import TensorBoardLogger


class BaseTrainer:
    def __init__(self, runner: BaseRunner, algorithm: BaseAlgorithm, logger: TensorBoardLogger):
        self.runner = runner
        self.algorithm = algorithm
        self.logger = logger

    def train(self):
        for i in range(100_000):
            self.runner.run()
            stats = self.algorithm.update(self.runner.agent, self.runner.buffer)
            self.logger.log_metrics(stats, i)