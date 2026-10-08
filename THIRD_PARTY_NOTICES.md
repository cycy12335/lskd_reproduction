# Source provenance

`models/architectures/resnet.py` is copied without architectural changes from:
https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/mdistiller/models/cifar/resnet.py

Upstream commit: eb569ac3783f2eed16fbc2b021101175af4fb2f2.
The official project credits mdistiller, Multi-Level-Logit-Distillation, CTKD,
and tiny-transformers. No root license file was present in this official checkout;
this provenance notice does not assign a new license to upstream code.

KD loss and CIFAR-100 normalization settings are implemented to match this same
commit's `mdistiller/distillers/KD.py` and `mdistiller/dataset/cifar100.py`.
The published ResNet32x4 teacher comes from the mdistiller checkpoints release.
