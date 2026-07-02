"""
Initial tensors for tensor network calculations.

Tensor leg convention:
- xi: Left (incoming from left neighbor)
- xo: Right (outgoing to right neighbor)
- yi: Up (incoming from above neighbor)
- yo: Down (outgoing to below neighbor)
"""

import cytnx
import numpy as np
import itertools


# =============================================================================
# 2D Ising Model Energy and Boltzmann Weight
# =============================================================================

def index_to_spin(i):
    """
    Map tensor index to Ising spin value.

    Args:
        i: Index (0 or 1)

    Returns:
        Spin value (+1 or -1)

    Mapping:
        0 -> +1 (spin up)
        1 -> -1 (spin down)
    """
    return 1 - 2 * i


def ising_bond_energy(i, j, J=1.0):
    """
    Compute bond energy for 2D Ising model.

    Args:
        i: Index of first spin (0 or 1)
        j: Index of second spin (0 or 1)
        J: Coupling constant (default: 1.0, ferromagnetic)

    Returns:
        Bond energy E = -J * σ_i * σ_j

    Energy values:
        E(0,0) = E(1,1) = -J  (aligned spins, low energy)
        E(0,1) = E(1,0) = +J  (anti-aligned spins, high energy)
    """
    sigma_i = index_to_spin(i)
    sigma_j = index_to_spin(j)
    return -J * sigma_i * sigma_j


def ising_boltzmann_weight(i, j, beta, J=1.0):
    """
    Compute Boltzmann weight for a bond in 2D Ising model.

    Args:
        i: Index of first spin (0 or 1)
        j: Index of second spin (0 or 1)
        beta: Inverse temperature (β = 1/kT)
        J: Coupling constant (default: 1.0)

    Returns:
        Boltzmann weight W = exp(-β * E) = exp(β * J * σ_i * σ_j)

    Weight values:
        W(0,0) = W(1,1) = exp(+βJ)  (aligned, favored)
        W(0,1) = W(1,0) = exp(-βJ)  (anti-aligned, suppressed)
    """
    E = ising_bond_energy(i, j, J)
    return np.exp(-beta * E)


def ising_boltzmann_matrix(beta, J=1.0):
    """
    Construct the 2x2 Boltzmann weight matrix for 2D Ising model.

    Args:
        beta: Inverse temperature
        J: Coupling constant (default: 1.0)

    Returns:
        2x2 numpy array W[i,j] = exp(-β * E(i,j))

    Matrix form:
        W = [[exp(+βJ), exp(-βJ)],
             [exp(-βJ), exp(+βJ)]]

        = [[cosh(βJ) + sinh(βJ), cosh(βJ) - sinh(βJ)],
           [cosh(βJ) - sinh(βJ), cosh(βJ) + sinh(βJ)]]
    """
    W = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            W[i, j] = ising_boltzmann_weight(i, j, beta, J)
    return W


def ising_boltzmann_matrix_sqrt(beta, J=1.0):
    """
    Construct the square root factorization of Boltzmann weight matrix.

    The Boltzmann weight matrix W can be factorized as:
        W[i,j] = Σ_s M[s,i] * M[s,j]

    where M is a 2x2 matrix obtained from eigendecomposition.

    Args:
        beta: Inverse temperature
        J: Coupling constant (default: 1.0)

    Returns:
        2x2 numpy array M such that W = M^T @ M

    Matrix form:
        M = [[√cosh(βJ), √sinh(βJ)],
             [√cosh(βJ), -√sinh(βJ)]]

    This factorization enables the tensor network representation
    where each site tensor is formed by contracting four M matrices.
    """
    c = np.sqrt(np.cosh(beta * J))
    s = np.sqrt(np.sinh(beta * J))
    M = np.array([[c, s],
                  [c, -s]])
    return M


# =============================================================================
# Tensor Constructors
# =============================================================================

def ising_tensor(beta, J=1.0):
    """
    Create initial tensor for 2D classical Ising model.
    Legs: xi (left), xo (right), yi (up), yo (down)

    Args:
        beta: Inverse temperature (β = 1/kT)
        J: Coupling constant (default: 1.0, ferromagnetic)

    Tensor diagram:

                  yi
                  │
                  ▼
                  │
            ┌─────┴─────┐
            │           │
    xi ──▶──┤     T     ├──▶── xo
            │           │
            └─────┬─────┘
                  │
                  ▼
                  │
                  yo

    The tensor is constructed from the Boltzmann weight matrix factorization:
        T[xi,xo,yi,yo] = Σ_s M[s,xi] * M[s,xo] * M[s,yi] * M[s,yo]
    where M is the square root of the Boltzmann weight matrix.
    """
    # Get square root factorization of Boltzmann weight matrix
    M = ising_boltzmann_matrix_sqrt(beta, J)

    # Create UniTensor with labeled legs
    bd = cytnx.Bond(2)
    T = cytnx.UniTensor([bd, bd, bd, bd], rowrank=2)
    T.set_name("T")
    T.relabel_(["xi", "xo", "yi", "yo"])

    # Fill tensor elements
    for xi, xo, yi, yo in itertools.product([0, 1], repeat=4):
        T.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value = (
            M[0, xi] * M[0, xo] * M[0, yi] * M[0, yo] +
            M[1, xi] * M[1, xo] * M[1, yi] * M[1, yo]
        )

    return T


def ising_tensor_directional(beta):
    """
    Create initial tensor for 2D classical Ising model with directional bonds.
    Legs: xi (left, IN), xo (right, OUT), yi (up, IN), yo (down, OUT)

    Tensor diagram:

                  yi
                  │
                  ▼
                  │
            ┌─────┴─────┐
            │           │
    xi ──▶──┤     T     ├──▶── xo
            │           │
            └─────┬─────┘
                  │
                  ▼
                  │
                  yo

    Bond directions:
        - xi: BD_IN  (incoming)
        - xo: BD_OUT (outgoing)
        - yi: BD_IN  (incoming)
        - yo: BD_OUT (outgoing)
    """
    # Boltzmann weight matrix
    M = np.array([[np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))],
                  [np.sqrt(np.cosh(beta)), -np.sqrt(np.sinh(beta))]])

    # Create directional bonds
    bd_in = cytnx.Bond(2, cytnx.BD_IN)
    bd_out = cytnx.Bond(2, cytnx.BD_OUT)

    # Create UniTensor with directional bonds: [xi, xo, yi, yo]
    T = cytnx.UniTensor([bd_in, bd_out, bd_in, bd_out], rowrank=2)
    T.set_name("T")
    T.set_labels(["xi", "xo", "yi", "yo"])

    # Fill tensor elements
    for xi, xo, yi, yo in itertools.product([0, 1], repeat=4):
        T.at(["xi", "xo", "yi", "yo"], [xi, xo, yi, yo]).value = (
            M[0, xi] * M[0, xo] * M[0, yi] * M[0, yo] +
            M[1, xi] * M[1, xo] * M[1, yi] * M[1, yo]
        )

    return T


def ising_tensor_z2(beta):
    """
    Create initial tensor for 2D classical Ising model with Z2 symmetry.
    Legs: xi (left, IN), xo (right, OUT), yi (up, IN), yo (down, OUT)

    Tensor diagram:

                  yi
                  │
                  ▼
                  │
            ┌─────┴─────┐
            │           │
    xi ──▶──┤     T     ├──▶── xo
            │           │
            └─────┬─────┘
                  │
                  ▼
                  │
                  yo

    Z2 Symmetry:
        - Each bond carries Z2 quantum numbers [0, 1] (even/odd parity)
        - Only blocks with total parity = 0 (mod 2) are non-zero
        - Block structure: (0,0,0,0), (0,0,1,1), (0,1,0,1), (0,1,1,0),
                          (1,0,0,1), (1,0,1,0), (1,1,0,0), (1,1,1,1)
    """
    # Boltzmann weights for each parity sector
    w_even = np.cosh(beta)  # weight for even parity (all same spin)
    w_odd = np.sinh(beta)   # weight for odd parity (mixed spins)

    # Z2 symmetry
    z2 = cytnx.Symmetry.Zn(2)

    # Create symmetric bonds with quantum numbers [0, 1], each with dim 1
    # IN bonds for xi, yi; OUT bonds for xo, yo
    bd_in = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [z2])
    bd_out = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [z2])

    # Create symmetric UniTensor
    T = cytnx.UniTensor([bd_in, bd_out, bd_in, bd_out], rowrank=2)
    T.set_name("T")
    T.set_labels(["xi", "xo", "yi", "yo"])

    # Fill non-zero blocks
    # For Z2 symmetry: xi + xo + yi + yo = 0 (mod 2) for incoming - outgoing
    # With BD_IN contributing +q and BD_OUT contributing -q:
    # Valid blocks satisfy: q_xi - q_xo + q_yi - q_yo = 0 (mod 2)

    # Get all valid blocks and fill them
    for blk_idx in range(T.Nblocks()):
        qnums = T.get_qindices(blk_idx)
        # qnums is list of [q_xi, q_xo, q_yi, q_yo]
        q_xi, q_xo, q_yi, q_yo = [q[0] for q in qnums]

        # Count parity: number of odd quantum numbers
        parity = q_xi + q_xo + q_yi + q_yo

        if parity == 0:
            # All even: (0,0,0,0)
            T.get_block_(blk_idx)[0] = w_even
        elif parity == 4:
            # All odd: (1,1,1,1)
            T.get_block_(blk_idx)[0] = w_odd
        else:
            # Mixed: 2 even, 2 odd
            T.get_block_(blk_idx)[0] = np.sqrt(w_even * w_odd)

    return T
