# Environment setup verification

Date: 2026-10-08 (Asia/Shanghai).
Project root: `D:\reproduction`.

## Configured environment

- Project-local `.venv`, CPython 3.10.20.
- PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126.
- CUDA runtime 12.6; cuDNN version 91002.
- NVIDIA driver 592.82.
- Actual GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM.
- The source plan assumes 8 GB; actual hardware is 6 GB. Determine a suitable
  batch size during future pipeline validation and record any deviation from the paper.
- Original user-provided plan is preserved in `PROJECT_PLAN.md`.

## Completed validation

- `python -m pip check`: No broken requirements found.
- `python scripts/verify_environment.py --device cuda --output reports/environment/local.json`:
  dependency imports, CUDA matrix multiplication, convolution forward and torchvision NMS passed.
- `python -m pytest -q`: **2 passed in 6.75s**.
  Tests exercise the compiled torchvision CPU operator and a tiny CPU convolution backward pass.
- Python and PowerShell scripts pass syntax checks; environment and workflow YAML parse successfully.
- Git exclusions checked for dataset, checkpoint, virtual environment and local interpreter paths.
  Teacher README remains eligible for version control.

Exact installed dependencies: `requirements-lock-windows-gpu.txt`.
Machine-readable validation results: `reports/environment/local.json`.
Recreate with: `powershell -File scripts/setup_environment.ps1` (requires uv).
The Conda environment specification is provided as an alternative; it was not instantiated locally.

## Pending work and boundaries

- No dataset or teacher checkpoint has been downloaded.
- No teacher evaluation or formal CIFAR-100 experiment has been run.
- Experiment registry contains the header only; no experiment IDs have been consumed.
- The full Baseline/KD/LS-KD pipeline smoke test is not implemented. Gate 0 remains incomplete.
- `.github/workflows/ci.yml` installs CPU wheels and runs environment checks/unit tests only.
  GitHub Actions has not been executed remotely. Formal GPU experiments must always run locally.
- Local Git repository initialized on `main`; no commit or remote push performed.

PyTorch package pairing and CUDA index reference:
https://pytorch.org/get-started/previous-versions/
