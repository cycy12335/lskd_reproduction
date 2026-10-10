# INC_001：CIFAR-100 下载中断

日期：2026-10-08（Asia/Shanghai）。

关联实验：无，属于正式实验前的准备阶段，未占用正式实验 ID。

状态：已解决。严重程度：中等。

## 预期与实际

预期获得完整的官方 CIFAR-100 归档，共 169,001,437 字节，MD5 为 `eb9058c3a382ffc7106e4002c42a8d85`，并通过 torchvision 的 train/test 文件完整性检查。

实际单流下载在 38,936,000 字节时中断，`scripts/prepare_data.py` 抛出 `RuntimeError: File not found or corrupted.`。该不完整文件未作为有效数据使用。

## 排查与处理

初步判断为网络或服务器传输中断。保留不完整文件，并尝试官方源分段下载。最终从已记录的 Hugging Face 镜像取得完整归档；仅在归档 MD5 与 torchvision 官方值完全一致，且 train/test 文件 MD5 均通过后接受数据。

完整来源、归档和分割文件校验信息见 `data/manifest.json`。此问题已解决，未覆盖模型 checkpoint 或正式实验记录。
