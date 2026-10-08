"""Verify dependencies and device operations without downloading data or training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone


def verify(device_name="cpu"):
    import torch
    import torchvision
    from torchvision.ops import nms

    package_names = ("torch", "torchvision", "numpy", "scipy", "pandas", "matplotlib", "PyYAML", "tqdm", "pytest")
    for name in ("numpy", "scipy", "pandas", "matplotlib", "yaml", "tqdm", "pytest"):
        __import__(name)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    with torch.inference_mode():
        a = torch.ones((16, 16), device=device)
        torch.testing.assert_close(a @ a, torch.full_like(a, 16.0))
        conv = torch.nn.Conv2d(3, 8, kernel_size=3).to(device)
        output = conv(torch.zeros((2, 3, 32, 32), device=device))
        assert tuple(output.shape) == (2, 8, 30, 30)
        assert bool(torch.isfinite(output).all())
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        assert nms(boxes, scores, 0.5).tolist() == [0]
    if device_name == "cuda":
        torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[1]
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_tested": device_name,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "nvidia_smi": driver,
        "checks": {"imports": "passed", "matmul": "passed", "convolution_forward": "passed", "torchvision_nms": "passed"},
        "cifar100_status": "not_verified",
        "cifar100_local_directory_exists": (root / "data/raw/cifar-100-python").is_dir(),
        "teacher_status": "not_verified",
        "pipeline_smoke_status": "not_implemented",
        "formal_experiments_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.device)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
