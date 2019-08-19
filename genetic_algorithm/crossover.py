import numpy as np
from typing import Tuple
from .individual import Individual

def simulated_binary_crossover(parent1: Individual, parent2: Individual, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This crossover is specific to floating-point representation.
    Simulate behavior of one-point crossover for binary representations.

    For large values of eta there is a higher probability that offspring will be created near the parents.
    For small values of eta, offspring will be more distant from parents

    Equation 9.9, 9.10, 9.11
    @TODO: Link equations
    """    
    # Calculate Gamma (Eq. 9.11)
    rand = np.random.random(parent1.chromosome.shape)
    gamma = np.empty(parent1.chromosome.shape)
    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # First case of equation 9.11
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Second case

    # Calculate Child 1 chromosome (Eq. 9.9)
    chromosome1 = 0.5 * ((1 + gamma)*parent1.chromosome + (1 - gamma)*parent2.chromosome)
    # Calculate Child 2 chromosome (Eq. 9.10)
    chromosome2 = 0.5 * ((1 - gamma)*parent1.chromosome + (1 + gamma)*parent2.chromosome)

    return chromosome1, chromosome2