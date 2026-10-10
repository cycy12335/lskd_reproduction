# EXP_002：续训检查与验证

日期：2026-10-09（Asia/Shanghai）。用户授权先尝试继续训练，无法恢复时才从头重训并覆盖记录。本次检查支持续训，因此无需触发从头重训分支。

## 恢复依据

暂停时完整保存到 epoch 120，latest checkpoint SHA256：
da6c25c11d1a53a31c7567c885aad3a3d22feaadf9dce3c17c3e20bb7d041287。

模型参数、SGD 动量、Torch/CUDA RNG 和训练 DataLoader generator 均已保存。恢复模型及优化器后，最后恢复 RNG，从 epoch 121 开始；学习率仍由原 epoch 公式计算，151/181/211 时衰减。epoch 121 的未完成 batch 不保留进度，重新执行整个 epoch；原日志中的未完成尝试保留，由 resume_from_epoch 标记分段。

旧 checkpoint 没有主进程 Python/NumPy RNG。当前数据增强在两个非持久 worker 内执行，每个 epoch 由已恢复的 DataLoader generator 重新生成 worker seed；主进程训练路径未使用这两个 RNG。测试集预处理无随机增强，独立测试 DataLoader generator 未保存不影响模型更新。此兼容依据仅针对当前代码，不代表任意训练程序都可按此恢复。

## 实现与 Gate

新增 operations/resume_kd.py，复用 training.trainer、training.evaluator、training.runner 的原 Gate 和 checkpoint 保存函数；模型、损失、数据加载与原训练实现未改动。原 source_fingerprint 仅覆盖其既定目录，新增入口不在其范围内，因此不能用该指纹代表完整续训程序；metadata.resume_sessions 和续训 checkpoint 另行记录入口 SHA256、Git 提交及恢复来源。

保留 Teacher EXP_000、CUDA smoke 和 Baseline Gate 的验证，不伪造指纹、不跳过检查。入口对 interrupted 状态、EXP ID、原配置、Teacher SHA256、checkpoint SHA256 和 epoch/metrics 一致性进行检查，使用互斥锁避免重复启动。

启动后先归档原实验目录与 registry，并在本地 checkpoint 目录内备份 best/latest；原 120 行指标不删改，续训追加第 121–240 行。继续同一个中断实验，不是复用 ID 创建新实验。原始训练提交保留，新增入口提交单独记录。

## 验证

执行 .venv/Scripts/python.exe -m pytest -q：14 项通过，25.60 秒。

新增两项测试使用真实项目的数据加载及训练函数，在带 BatchNorm/Dropout 的小模型上比较连续执行与序列化后重建恢复。CUDA 的指标、模型参数、SGD 动量和 RNG 完全一致；CPU 初次逐位比较出现约 1e-7 浮点差异，改为 rtol=1e-6、atol=1e-7 后通过，RNG 仍逐位相等。该结果验证了代表性恢复路径，不等价于宣称整个正式训练跨进程必然逐位复现。

实际 checkpoint 的 --check 通过，恢复边界 120→121，总预算仍为 240 epochs。启动后还需核对日志推进、首个续训 epoch 及 checkpoint。

## 后续边界

这是用户明确恢复训练后的操作，替代上一条暂停指令。KD 完成后先审查结果，LS-KD 尚未启动。遇到训练异常或结果警告时保存现场并产生中文 Incident，不自动进入下一阶段。

### 续训已启动：2026-10-09 08:06（UTC+8）

EXP_002 已于 08:06:04 启动续训，训练 PID 38520、启动包装 PID 33992。恢复自 epoch 120，从 epoch 121 继续；运行状态以实时 metadata 为准。新增入口提交 670c65c，SHA256 cd23133ae35e3f80fb73cd78cc31854ae05eff09d5ae2330fa219f681554f527。旧核心源码指纹仍为 b8c6fa0e7d8b14068f72b3538fc4712865375efa1fca1bc8eecba956c5ccf3ff。

归档：reports/progress/EXP_002_resume_20261009_000606；checkpoint 备份：models/checkpoints/kd/EXP_002/before_resume_20261009_000606。已验证前 120 行指标未改变，备份哈希匹配。恢复后 epoch 121 的 step 1、100 损失和准确率与暂停前相应日志显示值一致；此为恢复正确性的额外证据，不等价于完整训练结果保证。

启动 stdout/stderr：logs/system/kd_resume_20261009_080604.stdout.log 及 .stderr.log。锁已建立，GPU 正常执行。不要重复启动或改动运行中的源码。无需从头重训或覆盖原有历史记录。
