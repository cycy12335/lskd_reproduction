# EXP_006 续训验证（2026-10-10）

用户授权继续多种子训练。实时核查无正式训练进程，两个训练锁均不存在；队列 paused，EXP_006 interrupted，完成 60/240 epochs，阶段最佳 57.81%（epoch 50），最近 54.40%。这些不是最终结果。

恢复检查通过：原始配置、核心源码指纹、EXP_000 Teacher Gate、前 60 行指标、best/latest checkpoint 与暂停记录一致；EXP_004 完整 240 epochs 重新验收通过（best 72.67%、final 72.47%）。后续四项 ID 未使用。

latest SHA256：1d66426fac21283566d68ed932f55f894425bd9de9ddb0d67053d60e1dbb818b

best SHA256：c84d7a3c24ae1928e95cbdf643140a853b0862a13cf63108cf2a9bb2a84c6f0d

专用入口 operations/resume_exp006.py 保留已有 EXP_002 入口，核心训练源码未修改；入口与控制器另记哈希和 Git 版本。恢复模型、优化器、Torch/CUDA RNG、训练 DataLoader RNG。原检查点未保存主进程 Python/NumPy RNG；训练增强在由 loader RNG 重新播种的 worker 中执行，测试转换确定性。原记录和 checkpoint 启动时先归档，原训练 commit 保留，续训版本独立记录。

验证：队列 --resume --check 通过；pytest tests/test_multiseed_queue.py tests/test_resume.py 为 8 passed。测试覆盖控制器失败停止，以及 CPU/CUDA 的连续训练与断点恢复。首次 CPU 多线程复测在优化器一个元素出现 2.7e-7 差异；测试固定 CPU 单线程后通过，原容限未放宽。CUDA 仍要求参数、优化器、随机状态和指标完全一致并通过。该测试线程设置不改变正式训练。

恢复后队列：EXP_006 → EXP_008 → EXP_005 → EXP_007 → EXP_009。每项完成后执行原 Gate 和完整性验收；warning/失败保留现场并停止后续。启动和实际进度见实时 state.json、metadata 和日志。