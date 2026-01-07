from abc import ABC, abstractmethod
from typing import Any, Tuple
from networks import extractors, heads
from torch import nn, Tensor


class BaseArchitecture(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, observations:Tensor) -> Tuple[Any, Tensor]:
        pass


class SharedArchitecture(BaseArchitecture):
    def __init__(self, 
                 shared_extractor: extractors.Extractor, 
                 actor_head: heads.VectorHead, 
                 critic_head: heads.ScalarHead):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.actor_head = actor_head
        self.critic_head = critic_head

    
    def forward(self, observations: Tensor) -> Tuple[Any, Tensor]:
        features = self.shared_extractor(observations)
        return self.actor_head(features), self.critic_head(features)
    

class SeparatedArchitecture(BaseArchitecture):
    def __init__(self, 
                 actor_extractor: extractors.Extractor, 
                 actor_head: heads.VectorHead,
                 critic_extractor: extractors.Extractor,
                 critic_head: heads.ScalarHead):
        super().__init__()
        self.actor_extractor = actor_extractor
        self.actor_head = actor_head
        self.critic_extractor = critic_extractor
        self.critic_head = critic_head
    

    def forward(self, observations: Tensor) -> Tuple[Any, Tensor]:
        actor_features = self.actor_extractor(observations)
        critic_features = self.critic_extractor(observations)
        return self.actor_head(actor_features), self.critic_head(critic_features)
        