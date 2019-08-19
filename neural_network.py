import numpy as np
from typing import List


class FeedForward(object):
    def __init__(self,
                 num_inputs: int,
                 hidden_layers: List[int],
                 activations,
                 seed: None):
        