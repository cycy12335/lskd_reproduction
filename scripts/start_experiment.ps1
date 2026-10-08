# Starts one authorized formal experiment locally, hidden, with persistent logs.
param([ValidateSet('baseline','kd','lskd')][string]$Method = 'baseline')
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
Set-Location $project
if ($env:GITHUB_ACTIONS) { throw 'Formal GPU experiments are forbidden in GitHub Actions' }
if (Test-Path '.tools/active_experiment.lock') { throw 'An experiment lock exists; inspect the recorded PID before starting another job' }
New-Item -ItemType Directory -Path logs/system -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$python = Join-Path $project '.venv/Scripts/python.exe'
$stdout = Join-Path $project "logs/system/${Method}_${stamp}.stdout.log"
$stderr = Join-Path $project "logs/system/${Method}_${stamp}.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList @('-u', "scripts/train_$Method.py") -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = @{pid=$process.Id; method=$Method; started_at=(Get-Date -Format o); stdout=$stdout; stderr=$stderr}
$record | ConvertTo-Json | Set-Content -LiteralPath '.tools/last_launch.json' -Encoding UTF8
$record | ConvertTo-Json
