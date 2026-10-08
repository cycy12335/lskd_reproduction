## 处理结果：RESOLVED（2026-10-08）

用户明确本项目目标是方法论实验复现，无需精确复现论文历史实验。现有匹配参数的 KD/LS-KD 设计符合该目标，因此解除启动阻塞。关闭原因是研究范围澄清，不是历史参数已被证实。下文保留发现时的排查记录；INVESTIGATING 与启动阻塞描述均为历史状态。

按现有配置继续：T=2、CE=0.1、KD=9，仅切换标准化。论文 73.33% 不作为验收目标，不额外要求历史协议复现实验。Baseline Gate 2 已通过，允许按顺序启动 EXP_002 KD；EXP_003 仍须等待 KD 完成并审查。

# INC_002：KD 论文对照协议溯源待完成

日期：2026-10-08（Asia/Shanghai）。状态：INVESTIGATING。严重程度：中等（影响论文对照有效性，未发现运行故障）。关联：计划中的 EXP_002、EXP_003，均未启动；EXP_001 已完成且通过检查。

当前 HEAD：1c3e7ed；训练提交：7eb60bb7872a7f2085caa2e61fad1b035b76986c；固定官方提交：eb569ac3783f2eed16fbc2b021101175af4fb2f2。

## 预期与实际

预期：KD 启动前明确论文 Vanilla KD 73.33% 的参数来源，区分论文默认协议复现与匹配参数、仅切换标准化的对照。

实际：本地 KD/LS-KD 均为 T=2、CE=0.1、KD=9。固定官方 cfg.py 和 README 的 KD 启动方式支持这一当前默认配置；其 YAML 未覆盖蒸馏参数。官方 README 列出 KD 73.33%、KD+Ours 76.62%，但不能仅凭表格证明 73.33% 使用了当前默认参数。

mdistiller 现行 master/cfg.py 的 KD 默认值为 T=4、CE=0.1、KD=0.9；温度不同，KD 权重相差十倍。这是可变分支，尚未固定产生论文数字的历史提交或运行记录，不能直接认定为论文精确协议。RepDistiller 原作者仓库也列出 73.33%，是下一步溯源线索；相同数值不等于来源证明。

## 排查与现场保全

已核对本地固定上游 README.md、mdistiller/engine/cfg.py、configs/cifar100/kd/resnet32x4_resnet8x4.yaml、tools/train.py 以及本地配置。损失使用两个独立权重，KD 项含 T²，不能替换为权重和为 1 的 alpha 公式。

补充材料检索结果支持 LS 方法默认 T=2、CE=0.1、KD=9；正文及补充全文直接访问部分返回 403/404，历史 Vanilla 协议尚未完整核实。

已保存 Baseline 原始记录与 checkpoint 的 SHA256 清单：reports/validation/EXP_001_completion_audit_20261008.json。Baseline error.log 和启动 stderr 均为空，本 Incident 无训练错误栈。环境保持 Python 3.10.20、PyTorch 2.8.0+cu126、本地 RTX 3060 Laptop 6 GiB。未修改训练源码、配置或原始实验记录，未启动新训练。

## 解决条件与下一步

取得并固定 Vanilla KD 73.33% 对应代码提交、参数和表格来源，不能仅凭现行默认值归因。将 T=2/CE=0.1/KD=9 的匹配参数对照与历史默认协议复现分别报告。新增对照须协调 registry 和原计划、使用未用 ID，不覆盖已完成实验。

协议解决前不进入 Gate 3；本问题不否定 Baseline 的 Gate 2 验收。源码指纹仍匹配 EXP_000 和 smoke test；Teacher Gate 固定 EXP_000，后续若修改相关 Python 文件，须先支持新 Teacher 验证 ID，再按新 ID 重验，不覆盖 EXP_000、不跳过 Gate。报告模板也在源码指纹范围内。

## 来源

- [固定官方配置](https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/mdistiller/engine/cfg.py)
- [固定官方 README](https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/README.md)
- [mdistiller 现行配置](https://raw.githubusercontent.com/megvii-research/mdistiller/master/mdistiller/engine/cfg.py)，读取日期 2026-10-08，非历史证据。
- [原作者 RepDistiller](https://github.com/HobbitLong/RepDistiller)
- [论文补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Sun_Logit_Standardization_in_CVPR_2024_supplemental.pdf)