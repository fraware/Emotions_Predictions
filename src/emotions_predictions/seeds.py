from __future__ import annotations

import os
import random

import numpy as np


def set_random_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and TensorFlow.

    CPU runs are most reproducible; GPU ops may stay nondeterministic.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
