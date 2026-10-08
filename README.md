# Logit Standardization in Knowledge Distillation

CVPR 2024 CIFAR-100 reproduction. Root: `D:\reproduction`.
Teacher: ResNet32x4. Student: ResNet8x4. Local GPU: RTX 3060 Laptop, **6 GiB**.

## Environment

Python 3.10.20, PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126.
Use `.\.venv\Scripts\python.exe` to avoid the system Python 3.14.
Pinned packages: `requirements.txt`, `requirements-gpu.txt` and
`requirements-lock-windows-gpu.txt`. CPU CI uses `requirements-cpu.txt`.
Recreate with uv: `powershell -File scripts/setup_environment.ps1`.
Conda alternative: `conda env create -f environment.yml`.

## Local run order

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_smoke_test.py --device cpu --dataset fake --batch-size 4
.\.venv\Scripts\python.exe scripts/run_smoke_test.py --device cuda --dataset cifar100 --batch-size 64
# Commit all source and configuration before formal experiments.
.\.venv\Scripts\python.exe scripts/verify_teacher.py
powershell -File scripts/start_experiment.ps1 -Method baseline
```

Teacher must match 79.42% within 0.5 percentage points before EXP_001 starts.
Formal baseline is seed 42, 240 full epochs, batch 64, LR 0.05, SGD momentum 0.9,
weight decay 0.0005. Gate checks prevent running without a current full smoke test
or verified teacher. Inspect the completed baseline before advancing to KD/LS-KD.
Prepared KD/LS-KD configs match the pinned official release defaults T=2,
CE=0.1, KD=9 and differ only in standardization. Review the published vanilla-KD
comparison protocol before using them to claim reproduction of a paper table entry.
See `reports/analysis_protocol.md` for source references and declared deviations.

Use `--experiment-id EXP_XXX` for a new retry ID; completed or failed IDs cannot
be reused. Resume is not implemented. Failed experiments preserve evidence and
create incident reports. `.tools/active_experiment.lock` prevents concurrent jobs;
after a crash, confirm the recorded PID has exited before removing a stale lock.

## Results and tracking

`experiments/registry.csv`: experiment status and code commit.
Each experiment records config, metadata, metrics, train/error logs and summary.
Check `metadata.json` for current epoch, best accuracy and checkpoint SHA256.
`.tools/last_launch.json` contains the detached local process ID and console logs.
Large checkpoints stay in `models/checkpoints/`; teacher provenance is in
`models/pretrained/teacher/source.json`. Dataset and checkpoint binaries are ignored.
No paper accuracy or statistical conclusion is assumed in advance.

## GitHub Actions policy

GitHub Actions runs unit tests and tiny synthetic **CPU-only** smoke tests.
Never run formal CIFAR-100 GPU training in Actions, including self-hosted runners.
Source, configs, metrics, analysis and metadata belong in GitHub; checkpoint binaries
stay local. Workflows have not yet been executed remotely. No GitHub remote is configured.

Official code: https://github.com/sunshangquan/logit-standardization-KD
Official paper: https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html
Original management plan: `PROJECT_PLAN.md`.
