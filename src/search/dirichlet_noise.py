import numpy as np


def add_dirichlet_noise(probs, alpha=0.3, epsilon=0.25):
    """
    Adds exploration noise (used later for AlphaZero).
    """
    noise = np.random.dirichlet([alpha] * len(probs))
    return (1 - epsilon) * probs + epsilon * noise
