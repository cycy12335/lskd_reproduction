# Run from any directory: powershell -File scripts/setup_environment.ps1
# Requires uv. Installs only into this project; does not start experiments.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}
Invoke-Uv python install 3.10.20 --install-dir .tools/python --no-bin
$interpreter = Join-Path (Get-Location) '.tools/python/cpython-3.10.20-windows-x86_64-none/python.exe'
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    Invoke-Uv venv .venv --python $interpreter --seed
}
Invoke-Uv pip install --python .venv/Scripts/python.exe -r requirements-gpu.txt
Invoke-Uv pip install --python .venv/Scripts/python.exe -r requirements.txt
& ./.venv/Scripts/python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& ./.venv/Scripts/python.exe scripts/verify_environment.py --device cuda
if ($LASTEXITCODE -ne 0) { throw 'CUDA environment check failed' }
