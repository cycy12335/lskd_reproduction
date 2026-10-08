# Preflight validation

2026-10-08: 12 unit tests passed in 4.67 seconds.
Command: `.venv/Scripts/python.exe -m pytest -q`.

Full baseline/KD/LS-KD smoke tests passed on CPU with tiny synthetic data and on
local CUDA with real CIFAR-100 (2 batches of 64 train samples per method, followed
by 128-sample evaluation). Student weights updated, gradients were finite, teacher
remained frozen, and no CUDA OOM occurred. Peak allocated tensor memory was
374268928 bytes (this does not include all driver/runtime reserved memory).

Evidence: `smoke_cpu_fake.json`, `smoke_cuda_cifar100.json`.
Source fingerprint: b8c6fa0e7d8b14068f72b3538fc4712865375efa1fca1bc8eecba956c5ccf3ff.
Teacher strict state-dict load and dataset official MD5 checks passed.
Gate 0 is passed. Gate 1 still requires full Teacher test-set evaluation.
