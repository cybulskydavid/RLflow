import hydra
import torch
import numpy as np
import random
from omegaconf import DictConfig
from hydra.utils import instantiate
from configs.runner import OffPolicyRunnerConfig

from runners.off_policy_runner import OffPolicyRunner
from trainers.off_policy_trainer import OffPolicyTrainer
from algorithms.sac import SAC

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../config", config_name="sac")
def main(cfg: DictConfig):
    
    set_global_seed(cfg.seed)

    env = hydra.utils.instantiate(cfg.env)
    buffer = hydra.utils.instantiate(
        cfg.buffer,
        state_shape=env.spec.obs_shape,
        action_shape=env.spec.action_shape)
    
    target_entropy = -float(np.prod(env.spec.action_shape))
    print(f"Target Entropy: {target_entropy}")

    agent = hydra.utils.instantiate(
        cfg.agent,
        state_shape=env.spec.obs_shape[0],
        action_shape=env.spec.action_shape[0]
    )

    actor_optim = torch.optim.Adam(
        [{'params': agent.architecture.actor.parameters(), 'lr': 3e-4}],
    )
    
    critic_optim = torch.optim.Adam(
        [{'params': agent.architecture.critic1.parameters(), 'lr': 1e-3},
         {'params': agent.architecture.critic2.parameters(), 'lr': 1e-3}],
    )
    
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)

    algorithm = hydra.utils.instantiate(
        cfg.algorithm,
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
        alpha_optimizer=alpha_optim,
        target_entropy=target_entropy,
    )
    
    algorithm.log_alpha = log_alpha

    logger = hydra.utils.instantiate(cfg.logger)

    runner = hydra.utils.instantiate(
        cfg.runner,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger
    )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        runner=runner,
        algorithm=algorithm,
        logger=logger
    )

    trainer.train()

if __name__ == "__main__":
    main()