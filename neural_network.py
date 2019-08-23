import numpy as np
from typing import List, Callable, NewType, Optional


ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])

sigmoid = ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))
tanh = ActivationFunction(lambda X: np.tanh(X))
relu = ActivationFunction(lambda X: np.maximum(0, X))
leaky_relu = ActivationFunction(lambda X: np.where(X > 0, X, X * 0.01))
linear = ActivationFunction(lambda X: X)



class FeedForwardNetwork(object):
    def __init__(self,
                 layer_nodes: List[int],
                 hidden_activation: ActivationFunction,
                 output_activation: ActivationFunction,
                 init_method: Optional[str] = 'xavier',
                 seed: Optional[int] = None):
        self.params = {}
        self.layer_nodes = layer_nodes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.inputs = None
        self.out = None

        # Initialize weights
        for l in range(1, len(self.layer_nodes)):
            self.params['W' + str(l)] = np.random.randn(self.layer_nodes[l], self.layer_nodes[l-1]) * 0.01
            self.params['b' + str(l)] = np.random.randn(self.layer_nodes[l], 1) # np.zeros((self.layer_nodes[l], 1))
            # self.params['W' + str(l)] = np.ones((self.layer_nodes[l], self.layer_nodes[l-1]))
            self.params['A' + str(l)] = None
        
        
    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        A_prev = X
        L = len(self.layer_nodes) - 1  # len(self.params) // 2

        # Feed hidden layers
        for l in range(1, L):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_prev = self.hidden_activation(Z)
            self.params['A' + str(l)] = A_prev

        # Feed output
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A_prev) + b
        out = self.output_activation(Z)
        self.params['A' + str(L)] = out

        self.out = out
        return out

    def softmax(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) / np.sum(np.exp(X), axis=0)