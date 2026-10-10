# 多种子训练已按用户要求暂停

时间：2026-10-09 21:09（UTC+8）。控制器及整个训练进程树已停止，后续任务不会自动启动。

EXP_004（Baseline，seed 3407）已完成并通过队列检查：最佳 Top-1 72.67%，最终 72.47%。
EXP_006（KD，seed 3407）已中断：完成 60/240 epochs，阶段最佳 Top-1 57.81%，最近完成 epoch 的 Top-1 54.4%。这些不是最终 KD 结果。

checkpoint 可加载，最新 checkpoint epoch 为 60，与指标/metadata 一致性：True。哈希见本目录 checkpoint_audit.json。原元数据、队列和进程/锁快照已保存。用户主动暂停，不是训练故障。

其他四项尚未启动。保留 .tools/multiseed_stop_requested.json 防止自动推进。不得直接重新运行当前队列；恢复须等待用户授权，先检查恢复状态。现有 operations/resume_kd.py 专用于 EXP_002，不能直接用于 EXP_006。
