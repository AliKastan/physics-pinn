# Contributing to physics-pinn

Thank you for your interest in contributing! This project welcomes contributions of all kinds.

## Getting Started

```bash
git clone https://github.com/AliKastan/physics-pinn.git
cd physics-pinn
pip install -e ".[dev,app]"
pytest tests/ -v
```

## Development Workflow

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new functionality (see `tests/` for examples)
3. **Run the full test suite** before submitting: `make test`
4. **Submit a pull request** with a clear description of the changes

## Code Style

- Follow existing code patterns in the repository
- Use Google-style docstrings for classes and functions
- Add type hints to function signatures
- Keep lines under 100 characters where practical

## Adding a New Physical System

To add a new ODE/PDE system:

1. Create `src/models/your_system_pinn.py` with a class inheriting from `BasePINN` (for ODEs) or following the `HeatPINN`/`WavePINN` pattern (for PDEs)
2. Add the residual function to `src/physics/equations.py`
3. Add validation solver to `src/utils/validation.py`
4. Create `tests/test_your_system.py` with at minimum:
   - Forward pass shape test
   - Residual nonzero before training
   - Training reduces loss over 100 epochs
5. Add a config in `configs/`
6. Update `src/models/__init__.py`

## Reporting Issues

Please include:
- Python and PyTorch versions
- Minimal code to reproduce the issue
- Full error traceback

## Running Benchmarks

```bash
make benchmark       # quick (CI-level)
make benchmark-full  # thorough (for paper results)
```
