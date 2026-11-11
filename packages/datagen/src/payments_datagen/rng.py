"""Independent, reproducible random streams.

Each generation stage draws from its own named stream derived from the seed. Adding a draw in one
stage therefore never changes the numbers another stage sees, and the output does not depend on
the order stages run in.
"""

import numpy as np

STREAMS = {
    "merchants": 1,
    "accounts": 2,
    "bins": 3,
    "legit": 4,
    "hard_negatives": 5,
    "fraud": 6,
    "outcomes": 7,
    "ids": 8,
    "settlement": 9,
    "breaks": 10,
    "shape": 11,
}


def stream(seed: int, name: str) -> np.random.Generator:
    """Return the generator for the stage ``name``; unknown names raise ``KeyError``."""
    return np.random.default_rng(np.random.SeedSequence([seed, STREAMS[name]]))
