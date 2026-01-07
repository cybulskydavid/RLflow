import torch
from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner


class BaseTrainer:
    def __init__(self, runner: BaseRunner, algorithm: BaseAlgorithm):
        self.runner = runner
        self.algorithm = algorithm

    
    def train(self):
        for i in range(100_000):
            self.runner.run()
            stats = self.algorithm.update(self.runner.agent, self.runner.buffer)
            print(i)
            print(stats)