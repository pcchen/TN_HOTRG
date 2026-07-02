"""TN_HOTRG: a generic HOTRG framework built on cytnx.

The HOTRG algorithm itself (`hotrg_cytnx.hotrg_step`) is still a stub; this package
currently provides the initial tensor constructors for 2D classical spin models.
"""

from tn_hotrg.initial_tensors import (
    ising_tensor,
    ising_tensor_directional,
    ising_tensor_z2,
)

__version__ = "0.1.0"

__all__ = [
    "ising_tensor",
    "ising_tensor_directional",
    "ising_tensor_z2",
]
