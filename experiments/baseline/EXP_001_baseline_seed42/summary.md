# EXP_001 Student Baseline ? running

Snapshot time (UTC): 2026-10-08T07:11:34.754854+00:00
Status: running
Purpose: establish the ResNet8x4 student baseline before KD and LS-KD.
Configuration: seed 42, full CIFAR-100, FP32, 240 epochs, batch 64, SGD LR 0.05.
Code commit: 7eb60bb7872a7f2085caa2e61fad1b035b76986c
Completed epochs at snapshot: 8/240.
Best test accuracy at snapshot: 45.89% (intermediate result only).
Teacher verification: EXP_000 passed, Top-1 79.42%, Top-5 94.58%.

This summary is an initial running snapshot. See metadata.json and metrics.csv for
live progress; the trainer replaces this summary with the final report at completion.
Next step: inspect the completed baseline and incident status before advancing to KD.
