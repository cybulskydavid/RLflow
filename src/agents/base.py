from abc import ABC


class BaseAgent(ABC):

    def __init__(self, state_shape, action_shape, architecture):
        super().__init__()
        # self.architecture = architecture
        self.architecture = architecture(state_shape, action_shape)
    

    # @abstractmethod
    # def get_action(
    #     self, 
    #     state: np.ndarray, 
    #     deterministic: bool = False
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     pass


    # @abstractmethod
    # def evaluate_actions(
    #     self, 
    #     states: np.ndarray, 
    #     actions: np.ndarray
    # ) -> Tuple[Tensor, Tensor, Tensor]:
    #     pass