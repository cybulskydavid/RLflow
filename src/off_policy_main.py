import hydra
import torch
import numpy as np
import random
from omegaconf import DictConfig
from hydra.utils import instantiate

from agents.continuous_sac import SACAgent
from configs.env import GymEnvConfig
from configs.network import LinearConfig, ReLUConfig
from configs.runner import OffPolicyRunnerConfig

from buffers.replay_buffer import ReplayBuffer
from envs.gym_env import GymEnv
from networks.architectures import SACArchitecture
from networks.extractors import Extractor
from networks.heads import SACHead, ScalarHead
from runners.off_policy_runner import OffPolicyRunner
from trainers.off_policy_trainer import OffPolicyTrainer
from algorithms.sac import SAC

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    
    set_global_seed(42)

    env = GymEnv(id="BipedalWalker-v3")
    
    target_entropy = -float(np.prod(env.spec.action_shape))
    print(f"Target Entropy: {target_entropy}")

    buffer = ReplayBuffer(
        buffer_size=1000000,
        state_shape=env.spec.obs_shape,
        action_shape=env.spec.action_shape
    )

    layer_definitions = [LinearConfig(out_features=64), ReLUConfig(),
                         LinearConfig(out_features=64), ReLUConfig()]

    actor_extractor = Extractor(env.spec.obs_shape[0], layer_definitions=layer_definitions)
    actor_head = SACHead(actor_extractor.feature_dim, env.spec.action_shape[0])
    critic_extractor1 = Extractor(env.spec.obs_shape[0]+env.spec.action_shape[0], layer_definitions=layer_definitions)
    critic_head1 = ScalarHead(critic_extractor1.feature_dim)
    critic_extractor2 = Extractor(env.spec.obs_shape[0]+env.spec.action_shape[0], layer_definitions=layer_definitions)
    critic_head2 = ScalarHead(critic_extractor2.feature_dim)

    architecture = SACArchitecture(
        actor_extractor=actor_extractor,
        actor_head=actor_head,
        critic_extractor1=critic_extractor1,
        critic_head1=critic_head1,
        critic_extractor2=critic_extractor2,
        critic_head2=critic_head2
    )

    agent = SACAgent(architecture=architecture, state_shape=env.spec.obs_shape, action_shape=env.spec.action_shape)

    actor_optim = torch.optim.Adam(
        agent.architecture.actor.parameters(), 
        lr=3e-4
    )
    
    critic_optim = torch.optim.Adam(
        [{'params': agent.architecture.critic1.parameters(), 'lr': 1e-3},
         {'params': agent.architecture.critic2.parameters(), 'lr': 1e-3}],
    )
    
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)

    algorithm = SAC(
        actor_optimizer=actor_optim,
        critic_optimizer=critic_optim,
        alpha_optimizer=alpha_optim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        autotune=True,
        target_entropy=target_entropy,
        batch_size=256
    )
    
    algorithm.log_alpha = log_alpha

    runner = OffPolicyRunner(
        env=env,
        agent=agent,
        buffer=buffer,
        cfg=OffPolicyRunnerConfig()
    )

    trainer = OffPolicyTrainer(
        runner=runner,
        algorithm=algorithm,
        max_env_steps=1000000,
        warmup_steps=10000,
        log_freq=2000
    )

    trainer.train()

if __name__ == "__main__":
    main()