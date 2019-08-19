import numpy as np
from typing import List, Callable, NewType, Optional


ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])

sigmoid = ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))
tanh = ActivationFunction(lambda X: np.tanh(X))
relu = ActivationFunction(lambda X: np.maximum(0, X))
leaky_relu = ActivationFunction(lambda X: np.where(X > 0, X, X * 0.01))


class FeedForwardNetwork(object):
    def __init__(self,
                 num_inputs: int,
                 hidden_layers: List[int],
                 hidden_activation_type: ActivationFunction,
                 output_activation_type: ActivationFunction,
                 init_method: Optional[str] = 'xavier',
                 seed: Optional[int] = None):
        self.params = {}
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers

        L = len(self.hidden_layers)
        # Initialize weights
        for layer in hidde


    def feed_forward(self):
