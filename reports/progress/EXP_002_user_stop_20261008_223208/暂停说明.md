# EXP_002：用户要求暂停训练

状态：interrupted（按用户要求停止进程）。停止时间：2026-10-08 22:32（UTC+8）。

已完成 120/240 epochs。阶段最佳 Top-1：58.4%（epoch 117），最后完成 epoch 的 Top-1：53.02%。这些均为阶段结果，不是最终 KD 结果，不用于通过 Gate 3。

训练进程树已终止；指标、日志、best/latest checkpoint 保留。checkpoint 可读取，epoch 与 SHA256 记录于 reports/progress/EXP_002_user_stop_20261008_223208/checkpoint_audit.json。停止前后原 metadata 和锁快照也保存在该目录。

本次由用户主动要求暂停，不属于训练失败。当前没有断点续训入口；不能承诺从原进度恢复。后续训练须等待用户指示，不能自动重启 EXP_002 或启动 LS-KD。
