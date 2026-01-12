from omegaconf import DictConfig
import torch
import random

import numpy as np
import hydra
from configs import register_configs
from trainers.base_trainer import BaseTrainer

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../config", config_name="ppo")
def main(cfg: DictConfig):

    set_global_seed(cfg.seed)

    env = hydra.utils.instantiate(cfg.env)

    agent = hydra.utils.instantiate(cfg.agent, state_shape=env.spec.obs_shape[0], action_shape=env.spec.action_shape[0])
    optimizer_cls = hydra.utils.instantiate(cfg.optimizer.cls)
    algorithm = hydra.utils.instantiate(cfg.algorithm, agent=agent, optimizer_cls=optimizer_cls, optimizer_params=cfg.optimizer.params)
    buffer = hydra.utils.instantiate(
        cfg.buffer,
        state_shape=env.spec.obs_shape, 
        action_shape=env.spec.action_shape)

    logger = hydra.utils.instantiate(cfg.logger)
    runner = hydra.utils.instantiate(cfg.runner, env=env, agent=agent, buffer=buffer, logger=logger)
    trainer = BaseTrainer(runner, algorithm, logger)
    trainer.train()

if __name__ == "__main__":
    register_configs()
    main()