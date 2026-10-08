import _bootstrap
import argparse
import gc
from pathlib import Path
import time
import torch
from models.architectures import build_model
from datasets.cifar100 import make_loader
from training.runtime import ROOT, seed_all, load_teacher, source_fingerprint, write_json, now
from training.trainer import train_epoch
from training.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dataset", choices=("fake", "cifar100"), default="fake")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    seed_all(42)
    device = torch.device(args.device)
    results = {}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for method in ("baseline", "kd", "lskd"):
        started = time.perf_counter()
        student = build_model("resnet8x4").to(device)
        teacher = None
        if method != "baseline":
            teacher = load_teacher(device) if args.dataset == "cifar100" else build_model("resnet32x4").to(device).eval().requires_grad_(False)
        config = {"method": method, "distillation": {"temperature": 2.0, "ce_weight": 0.1, "kd_weight": 9.0, "standardized": method == "lskd"}}
        loader = make_loader(ROOT / "data/raw", True, args.batch_size, workers=0, tiny=True, fake=args.dataset == "fake")
        test_loader = make_loader(ROOT / "data/raw", False, args.batch_size, workers=0, tiny=True, fake=args.dataset == "fake")
        optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
        before = student.fc.weight.detach().clone()
        metrics = train_epoch(student, teacher, loader, optimizer, config, device, check_gradients=True)
        assert not torch.equal(before, student.fc.weight)
        if teacher is not None:
            assert all(p.grad is None for p in teacher.parameters())
            assert not teacher.training
        metrics.update(evaluate(student, test_loader, device))
        metrics["seconds"] = time.perf_counter() - started
        results[method] = metrics
        del student, teacher, optimizer, before
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report = {"status": "passed", "checked_at": now(), "source_fingerprint": source_fingerprint(),
              "device": args.device, "dataset": args.dataset, "batch_size": args.batch_size, "results": results,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
    write_json(ROOT / f'reports/validation/smoke_{args.device}_{args.dataset}.json', report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
