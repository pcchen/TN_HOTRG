"""
HOTRG (Higher-Order Tensor Renormalization Group) implementation using cytnx library.

Tensor leg convention:
- xi: Left (incoming from left neighbor)
- xo: Right (outgoing to right neighbor)
- yi: Up (incoming from above neighbor)
- yo: Down (outgoing to below neighbor)
"""

import cytnx
import numpy as np
from tn_hotrg.initial_tensors import ising_tensor


def hotrg_step(T, chi):
    """
    Perform one HOTRG coarse-graining step.

    Args:
        T: Input tensor with legs [xi, xo, yi, yo]
        chi: Maximum bond dimension

    Returns:
        T_new: Coarse-grained tensor
    """
    # TODO: Implement HOTRG step
    # 1. Contract two tensors horizontally
    # 2. SVD truncation in x-direction
    # 3. Contract two tensors vertically
    # 4. SVD truncation in y-direction
    pass


def compute_free_energy(T, n_iter):
    """
    Compute free energy per site from tensor trace.

    Args:
        T: Tensor after n_iter HOTRG steps
        n_iter: Number of iterations performed

    Returns:
        Free energy per site
    """
    # Contract all legs to get trace
    T_trace = T.clone()
    T_trace.set_labels(["xi", "xo", "yi", "yo"])

    # Trace over x and y directions
    delta_x = cytnx.UniTensor(cytnx.eye(T.shape()[0]), rowrank=1)
    delta_x.set_labels(["xi", "xo"])

    delta_y = cytnx.UniTensor(cytnx.eye(T.shape()[2]), rowrank=1)
    delta_y.set_labels(["yi", "yo"])

    result = cytnx.Contract(T_trace, delta_x)
    result = cytnx.Contract(result, delta_y)

    trace_val = result.item()
    n_sites = 2 ** (2 * n_iter)

    return -np.log(trace_val) / n_sites


def main():
    # Parameters
    beta_c = 0.5 * np.log(1 + np.sqrt(2))  # Critical temperature of 2D Ising
    beta = beta_c
    chi = 16  # Bond dimension
    n_iter = 20  # Number of HOTRG iterations

    # Initialize tensor
    T = ising_tensor(beta)
    print(f"Initial tensor created with beta = {beta:.6f}")
    print(f"Bond dimension chi = {chi}")
    print(f"Number of iterations = {n_iter}")

    # Run HOTRG
    for i in range(n_iter):
        T = hotrg_step(T, chi)
        if T is None:
            print("HOTRG step not yet implemented")
            break

    # Compute free energy
    # f = compute_free_energy(T, n_iter)
    # print(f"Free energy per site: {f:.10f}")


if __name__ == "__main__":
    main()
