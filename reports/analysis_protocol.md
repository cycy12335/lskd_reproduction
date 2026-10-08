# Initial experimental protocol

This run starts with Teacher verification and the seed-42 Student Baseline only.
Follow-up KD and LS-KD are gated on completed, reviewed preceding experiments.

## Pinned sources

Official code: https://github.com/sunshangquan/logit-standardization-KD
Commit: eb569ac3783f2eed16fbc2b021101175af4fb2f2.
Official CIFAR-100 ResNet baseline config: `configs/cifar100/vanilla.yaml`.
Teacher checkpoint: mdistiller `checkpoints` release, `resnet32x4_vanilla/ckpt_epoch_240.pth`.

Baseline: ResNet8x4, SGD, 240 epochs, batch 64, LR 0.05, momentum 0.9,
weight decay 0.0005. LR drops by 0.1 starting at epochs 151, 181, 211,
matching official `epoch > stage` behavior. Crop 32/padding 4 and random horizontal
flip; CIFAR-100 normalization mean (0.5071, 0.4867, 0.4408), std (0.2675, 0.2565, 0.2761).

Teacher acceptance: expected 79.42% top-1 on the entire 10,000-image test set,
absolute deviation <= 0.5 percentage points. Full 50,000-image train split is used
for formal baseline training; no subset or shortened epoch count is allowed.
Baseline best accuracy < 70.5% raises an incident and stops gate advancement;
72.50% is the paper's reference student accuracy, not a guaranteed outcome.

## KD comparison caveat

The pinned official cfg.py currently has T=2, CE weight=0.1 and KD weight=9,
even for the nonstandardized KD command. Both prepared configs follow those
current defaults, with only standardization toggled, to support a single-variable
comparison. The original plan's alpha formula is not substituted for these independent
weights. Do not assume that the paper's 73.33% vanilla-KD table entry was produced
by this exact current release configuration. Before EXP_002, review the published
protocol and historical defaults; if a different comparison is necessary, document
it explicitly and keep the matched-toggle comparison separate from paper defaults.

## Declared deviations

RTX 3060 Laptop has 6 GiB rather than the plan's assumed 8 GB. Batch size stays 64
unless the full-size GPU smoke test proves otherwise; any change needs a new config.
Python 3.10.20/PyTorch 2.8.0 rather than upstream Python 3.8/PyTorch 1.7.0.
FP32, no AMP, no TF32. Deterministic algorithms on, cuDNN benchmark off;
upstream sets cuDNN benchmark on. Seed 42 is the project seed; paper reports
averages over four trials, while the project initially plans three. A single run
must not be presented as statistical confirmation of the paper's mean result.

## Operations

GitHub Actions runs CPU unit tests and tiny synthetic smoke tests only.
Formal GPU jobs run locally. Experiment folders are never reused, checkpoints stay
under models/checkpoints, and metadata includes code commit, config digest, teacher
digest, environment, metrics and best-checkpoint digest. No resume entry point is
currently provided; a failed run is preserved and a retry needs a new EXP ID.
