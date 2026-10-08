# INC_001: CIFAR-100 download truncated

Date: 2026-10-08 (Asia/Shanghai)
Related experiment: none; preflight, no formal experiment ID reserved.
Status: RESOLVED
Severity: Medium

Expected: complete official CIFAR-100 archive (169001437 bytes), MD5
`eb9058c3a382ffc7106e4002c42a8d85`, plus torchvision train/test integrity checks.
Actual: the official single-stream download ended at 38936000 bytes and raised
`RuntimeError: File not found or corrupted.` in `scripts/prepare_data.py`.

Initial diagnosis: network/server transfer interruption; no dataset has been accepted.
Actions: keep partial file, retry official host using checked byte ranges; verify the
assembled archive MD5 before extraction. Formal experiments remain blocked until
integrity checks and full GPU smoke test pass. No checkpoint or experiment was overwritten.

Resolution: a byte-equivalent archive from the recorded Hugging Face mirror passed official archive MD5 and train/test-file MD5 checks. Dataset accepted; see data/manifest.json.
