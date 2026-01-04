from algorithms.base_algorithm import BaseAlgorithm
from runners.base_runner import BaseRunner


class BaseTrainer:
    def __init__(self, runner: BaseRunner, algorithm: BaseAlgorithm):
        self.runner = runner
        self.algorithm = algorithm

    
    def train():
        pass