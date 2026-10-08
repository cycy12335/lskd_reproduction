import csv
import json
import os
from pathlib import Path
import time
import traceback
import yaml
import torch
from models.architectures import build_model
from datasets.cifar100 import make_loader
from .evaluator import evaluate
from .trainer import train_epoch
from .runtime import (ROOT, create_experiment, write_json, sha256, teacher_path, load_teacher,
                      seed_all, source_fingerprint, registry, incident, now)


def ensure_gates(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Formal experiments require the local CUDA GPU")
    if os.environ.get("GITHUB_ACTIONS"):
        raise RuntimeError("Formal experiments are forbidden in GitHub Actions")
    smoke = json.loads((ROOT / "reports/validation/smoke_cuda_cifar100.json").read_text())
    if smoke["status"] != "passed" or smoke["source_fingerprint"] != source_fingerprint():
        raise RuntimeError("A current full CUDA CIFAR-100 smoke test must pass first")
    if smoke["batch_size"] != config["training"]["batch_size"]:
        raise RuntimeError("Smoke test must validate the configured training batch size")
    if config["method"] == "teacher":
        return
    candidates = list((ROOT / "experiments/teacher").glob("EXP_000_*/metadata.json"))
    if len(candidates) != 1:
        raise RuntimeError("Teacher verification EXP_000 is required")
    teacher_info = json.loads(candidates[0].read_text())
    if (teacher_info["status"] != "completed" or
            teacher_info["teacher_sha256"] != sha256(teacher_path()) or
            teacher_info["source_fingerprint"] != source_fingerprint()):
        raise RuntimeError("Teacher verification is not valid for this source/checkpoint")
    dependencies = {"kd": "baseline", "lskd": "kd"}
    if config["method"] in dependencies:
        previous = dependencies[config["method"]]
        passed = False
        for path in (ROOT / "experiments" / previous).glob("*/metadata.json"):
            info = json.loads(path.read_text())
            prior_config = yaml.safe_load(path.with_name("config.yaml").read_text(encoding="utf-8"))
            matching = (info["status"] == "completed" and info["seed"] == config["seed"] and
                        info["source_fingerprint"] == source_fingerprint() and
                        prior_config["student"] == config["student"] and
                        prior_config["training"] == config["training"])
            if config["method"] == "lskd":
                matching = matching and prior_config["teacher"] == config["teacher"]
                matching = matching and all(prior_config["distillation"][key] == config["distillation"][key]
                                            for key in ("temperature", "ce_weight", "kd_weight"))
                matching = matching and info["teacher_sha256"] == sha256(teacher_path())
            if matching:
                passed = True
        if not passed:
            raise RuntimeError(f"Completed {previous} with the same seed is required")


def atomic_checkpoint(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def run(config_path, expected_method=None, experiment_id=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if expected_method and config["method"] != expected_method:
        raise ValueError("Config method does not match this entry point")
    if experiment_id:
        config["experiment_id"] = experiment_id
    seed_all(config.get("seed", 42))
    ensure_gates(config)
    lock_path = ROOT / ".tools/active_experiment.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid": os.getpid(), "experiment_id": config["experiment_id"], "started": now()}))
    folder = info = None
    try:
        folder, info = create_experiment(config)
        info["pid"] = os.getpid()
        info["teacher_sha256"] = sha256(teacher_path())
        info["dataset"] = {"train": 50000, "test": 10000, "integrity": "torchvision CIFAR100 MD5 verified at load"}
        info["config_sha256"] = sha256(folder / "config.yaml")
        write_json(folder / "metadata.json", info)
        def log(message):
            line = f'{now()} {message}'
            print(line, flush=True)
            with (folder / "train.log").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        device = torch.device("cuda")
        settings = config["training"]
        test_loader = make_loader(ROOT / "data/raw", False, settings["batch_size"], workers=1, seed=config.get("seed", 42))
        if config["method"] == "teacher":
            teacher = load_teacher(device)
            result = evaluate(teacher, test_loader, device)
            expected = config["expected_accuracy"]
            passed = abs(result["test_accuracy"] - expected) <= config["tolerance_pp"]
            info.update(result, status="completed" if passed else "warning", end_time=now(),
                        expected_accuracy=expected, tolerance_pp=config["tolerance_pp"])
            with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(result))
                writer.writeheader()
                writer.writerow(result)
            if not passed:
                incident(folder, config, info, str(result), f'Teacher accuracy {expected} +/- {config["tolerance_pp"]} pp')
            log(f'Teacher verification: {result}; status={info["status"]}')
            (folder / "summary.md").write_text(f'# Teacher verification\n\nExpected: {expected}%\n\nActual: {result["test_accuracy"]}%\n\nStatus: {info["status"]}\n', encoding="utf-8")
            registry(config, info, result["test_accuracy"])
            write_json(folder / "metadata.json", info)
            return info
        student = build_model(config["student"]).to(device)
        teacher = load_teacher(device) if config["method"] != "baseline" else None
        loader = make_loader(ROOT / "data/raw", True, settings["batch_size"], workers=settings["num_workers"], seed=config["seed"])
        optimizer = torch.optim.SGD(student.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"], weight_decay=settings["weight_decay"])
        best = -1.0
        checkpoint_dir = ROOT / "models/checkpoints" / config["method"] / config["experiment_id"]
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        fields = ["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "learning_rate", "train_ce", "train_kd", "test_top5_accuracy", "epoch_seconds", "peak_gpu_memory_bytes"]
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, settings["epochs"] + 1):
            started = time.perf_counter()
            lr = settings["learning_rate"] * settings["lr_decay_rate"] ** sum(epoch > stage for stage in settings["lr_decay_stages"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            log(f'epoch={epoch}/{settings["epochs"]} lr={lr}')
            metrics = train_epoch(student, teacher, loader, optimizer, config, device, log)
            metrics.update(evaluate(student, test_loader, device))
            metrics.update(epoch=epoch, learning_rate=lr, epoch_seconds=time.perf_counter() - started,
                           peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
            with (folder / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(metrics)
            state = {"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "config": config, "git_commit": info["git_commit"],
                     "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                     "loader_rng": loader.generator.get_state()}
            if metrics["test_accuracy"] > best:
                best = metrics["test_accuracy"]
                info["best_epoch"] = epoch
                best_path = checkpoint_dir / f'{config["experiment_id"]}_best.pth'
                atomic_checkpoint(best_path, state)
                info["best_checkpoint"] = str(best_path.relative_to(ROOT).as_posix())
                info["best_checkpoint_sha256"] = sha256(best_path)
            state["best_accuracy"] = best
            atomic_checkpoint(checkpoint_dir / f'{config["experiment_id"]}_latest.pth', state)
            info.update(last_epoch=epoch, best_accuracy=best, last_metrics=metrics, updated_at=now())
            write_json(folder / "metadata.json", info)
            registry(config, info, best)
            log(f'epoch_complete={epoch} test_accuracy={metrics["test_accuracy"]:.3f} best={best:.3f} epoch_seconds={metrics["epoch_seconds"]:.1f}')
        info.update(status="completed", end_time=now())
        if config["method"] == "baseline" and best < config["minimum_expected_accuracy"]:
            info["status"] = "warning"
            incident(folder, config, info, f"Baseline best accuracy={best}", f'Baseline accuracy >= {config["minimum_expected_accuracy"]}%')
        if config["method"] in ("kd", "lskd"):
            previous = "baseline" if config["method"] == "kd" else "kd"
            previous_values = [json.loads(p.read_text()) for p in (ROOT / "experiments" / previous).glob("*/metadata.json")]
            previous_best = max(p["best_accuracy"] for p in previous_values if p["status"] == "completed" and p["seed"] == config["seed"])
            if best <= previous_best:
                info["status"] = "warning"
                incident(folder, config, info, f'{config["method"]}={best}, {previous}={previous_best}', f'{config["method"]} > {previous}')
        (folder / "summary.md").write_text(f'# {config["experiment_id"]}\n\nMethod: {config["method"]}\n\nBest accuracy: {best}%\n\nFinal accuracy: {metrics["test_accuracy"]}%\n\nStatus: {info["status"]}\n\nNext step: review metrics and incidents before the next gate.\n', encoding="utf-8")
        registry(config, info, best)
        write_json(folder / "metadata.json", info)
        return info
    except BaseException as error:
        if folder is not None:
            details = traceback.format_exc()
            (folder / "error.log").write_text(details, encoding="utf-8")
            info.update(status="failed", end_time=now())
            incident(folder, config, info, f'{type(error).__name__}: {error}', "Finite losses, valid checkpoint and complete experiment")
            registry(config, info, info.get("best_accuracy", ""))
            write_json(folder / "metadata.json", info)
            (folder / "summary.md").write_text(f'# Failed experiment\n\n{error}\n\nSee error.log and incident report.\n', encoding="utf-8")
        raise
    finally:
        lock_path.unlink()
