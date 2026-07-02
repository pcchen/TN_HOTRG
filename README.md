# TN_HOTRG

A generic HOTRG (Higher-Order Tensor Renormalization Group) framework built on the
[cytnx](https://github.com/Cytnx-dev/Cytnx) tensor-network library.

> Status: project scaffold only — the HOTRG algorithm is not implemented yet.

## Requirements

- [uv](https://docs.astral.sh/uv/) (used for environment and dependency management)
- Python >= 3.10 (uv will download a suitable interpreter automatically)
- cytnx >= 1.1.0 (installed as a project dependency)

## Set up the environment

Clone the repository, then let uv create the virtual environment and install all
dependencies from the lockfile:

```bash
git clone <repository-url>
cd TN_HOTRG
uv sync
```

`uv sync` creates a `.venv/` in the project root and installs the pinned dependencies
(`uv.lock`). No manual `pip install` or `venv` activation is required.

## Verify the setup

```bash
uv run python -c "import cytnx; print('cytnx', cytnx.__version__)"
uv run python -c "import tn_hotrg; print('tn_hotrg', tn_hotrg.__version__)"
```

Expected output:

```
cytnx 1.1.0
tn_hotrg 0.1.0
```

## Running code

Prefix commands with `uv run` to execute them inside the project environment, e.g.:

```bash
uv run python your_script.py
```

Alternatively, activate the environment once per shell session:

```bash
source .venv/bin/activate
```
