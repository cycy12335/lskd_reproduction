# ResNet32x4 CIFAR-100 teacher

Source: https://github.com/megvii-research/mdistiller/releases/tag/checkpoints
Archive: `cifar_teachers.tar`, member `cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth`.
Local file: `teacher_resnet32x4_cifar100.pth` (ignored by Git).
Expected test top-1 accuracy: 79.42%, as reported in the LS-KD paper.
Checkpoint SHA256: `22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f`.

Full source, retrieval timestamp and archive digest: `source.json` (tracked).
Digests are locally calculated for traceability, not independently published checksums.
Load uses `weights_only=True` with only required NumPy scalar/dtype allowlists.
Strict state-dict loading has passed. Full test-set verification is required separately.
