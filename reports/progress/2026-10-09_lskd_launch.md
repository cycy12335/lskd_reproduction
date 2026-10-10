# LS-KD 启动记录

## LS-KD 已启动：2026-10-09 10:17（UTC+8）

EXP_003 LS-KD 已于 10:17:08 启动，训练 PID 23292、启动包装 PID 36392。seed 42、batch 64、240 epochs，T=2、CE=0.1、KD=9、standardized=true。与 KD 的配置仅方法标识、EXP ID 及标准化开关不同，核心源码指纹一致。提交：38fcf3f1806142ae203445fb359b846377c6f7c6。

实时记录：experiments/lskd/EXP_003_lskd_seed42/；锁：.tools/active_experiment.lock；启动记录：.tools/last_launch.json。stdout/stderr：logs/system/lskd_20261009_101708.stdout.log 及 .stderr.log。初始进程、GPU 和日志推进正常，错误输出为空。不得重复启动或修改训练逻辑；完成后先核查结果，再讨论多种子实验。

KD EXP_002 已完成并验收：最佳 73.34%，最终 72.64%，Gate 3 通过。原始指标、元数据和 checkpoint 不改写。旧的运行与暂停段落仅作历史溯源。
