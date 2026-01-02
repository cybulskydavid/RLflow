from typing import Any, List
from torch import nn, Tensor
from networks.builders import build_layer


class Extractor(nn.Module):
    def __init__(self, obs_dim: int, layer_definitions: List[Any]) -> None:
        super().__init__()

        layers = []
        input_dim = obs_dim

        for layer_definition in layer_definitions:
            layer, input_dim = build_layer(layer_definition, input_dim)
            layers.append(layer)


        self.model = nn.Sequential(*layers)
        self._feature_dim = input_dim


    def forward(self, observations: Tensor) -> Tensor:
        return self.model(observations)
    

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# Currently not needed
# class Extractor(nn.Module, ABC):

#     @property
#     def feature_dim(self) -> int:
#         pass

# class MLPExtractor():
#     def __init__(self, obs_dim: int, layer_definitions: MLPExtractorConfig) -> None:
#         super.__init__()

#         layers = []
#         input_dim = obs_dim

#         for layer_definition in layer_definitions:
#             layer, input_dim = build_layer(layer_definition, input_dim)
#             layers.append(layer)


#         self.model = nn.Sequential(layers)
#         self._feature_dim = input_dim


#     def forward(self, x:Tensor) -> Tensor:
#         return self.model(x)
    

#     @property
#     def feature_dim(self) -> int:
#         return self._feature_dim
    


# class CNNExtractor():
#     def __init__(self, obs_dim: int, layer_definitions: CNNExtractorConfig):
#         pass