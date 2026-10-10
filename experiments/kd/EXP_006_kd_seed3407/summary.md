# EXP_006：KD 续训记录

状态：running。已完成 60/240 epochs。

最佳 Top-1：57.81%（epoch 50）。最近完成 epoch 的 Top-1：54.4%。

从 epoch 60 checkpoint 恢复，前 60 个 epoch 指标保留；续训来源见 metadata.json 的 resume_sessions。
未完成 240 epochs 时均为阶段结果。完成后检查完整结果与异常，再推进 LS-KD。
