import torch
from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner


class BaseTrainer:
    def __init__(self, runner: BaseRunner, algorithm: BaseAlgorithm):
        self.runner = runner
        self.algorithm = algorithm

    
    def train(self):
        for i in range(100_000):
            self.runner.agent.architecture.eval()
            with torch.no_grad():
                self.runner.run()
            self.runner.agent.architecture.train()
            stats = self.algorithm.update(self.runner.agent, self.runner.buffer)
            print(i)
            print(stats)