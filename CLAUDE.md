# TN_HOTRG

A generic HOTRG (Higher-Order Tensor Renormalization Group) framework for 2D classical
spin models, built on the `cytnx` tensor-network library (`cytnx>=1.1.0`).

## Environment
Managed by `uv` (see `README.md` for setup). Create/refresh the environment with:
```
uv sync
```

## Run Tests
```
uv run pytest          # or: uv run pytest -v
```

## Run the HOTRG driver
```
uv run python -m tn_hotrg.hotrg_cytnx
```

## Project Structure (src/ layout)
- `src/tn_hotrg/initial_tensors.py` - Tensor constructors for spin models (Ising, Z2)
- `src/tn_hotrg/hotrg_cytnx.py` - HOTRG algorithm (`hotrg_step()` is a TODO stub)
- `src/tn_hotrg/model.py` - Clock-type lattice model (Zn symmetry, mostly empty)
- `src/tn_hotrg/__init__.py` - Package root; re-exports the Ising tensor constructors
- `tests/test_initial_tensors.py` - pytest unit tests

## Tensor Leg Convention
All tensors use labeled legs: `xi` (left/in), `xo` (right/out), `yi` (up/in), `yo` (down/out)

## Key cytnx Patterns
- `cytnx.UniTensor([bonds], rowrank=2)` to create tensors
- `.set_labels([...])` / `.relabel_([...])` to set leg names
- `.at(["xi","xo","yi","yo"], [i,j,k,l]).value` to set/get elements
- `cytnx.Bond(dim, cytnx.BD_IN/BD_OUT)` for directional bonds
- `cytnx.Bond(cytnx.BD_IN, [[q0],[q1]], [dim0,dim1], [symmetry])` for symmetric bonds
- `cytnx.Symmetry.Zn(n)` for Zn symmetry
- `cytnx.Contract(A, B)` to contract tensors by matching labels

## Three Ising Tensor Variants
- `ising_tensor(beta)` - plain, no bond directions
- `ising_tensor_directional(beta)` - with BD_IN/BD_OUT on bonds
- `ising_tensor_z2(beta)` - with explicit Z2 quantum numbers (block-sparse)

## Known Issues
- `ising_tensor_z2()` and its tests (`TestIsingTensorZ2`) are marked `xfail`: cytnx 1.1.0's
  `get_qindices()` now returns a list of ints (not lists), so `initial_tensors.py` needs an
  update before the Z2 constructor works. Tracked pending fix.
- `hotrg_cytnx.hotrg_step()` is a TODO stub — the coarse-graining step is not yet implemented.
