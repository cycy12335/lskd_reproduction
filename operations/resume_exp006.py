"""Epoch-boundary continuation; core model/loss/training modules stay unchanged."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import csv
import json
import math
import shutil
import time
import traceback
import torch
import yaml
from datasets.cifar100 import make_loader
from models.architectures import build_model
from training.trainer import train_epoch
from training.evaluator import evaluate
from training.runner import ensure_gates, atomic_checkpoint
from training.runtime import (seed_all, source_fingerprint, sha256, load_teacher,
                              teacher_path, registry, write_json, now, git_commit,
                              check_code_clean)

FOLDER = ROOT / "experiments/kd/EXP_006_kd_seed3407"
CHECKPOINTS = ROOT / "models/checkpoints/kd/EXP_006"
STOP_AUDIT = ROOT / "reports/progress/multiseed_user_stop_20261009_210928/checkpoint_audit.json"


def restore_state(state, student, optimizer, loader, device):
    student.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    loader.generator.set_state(state["loader_rng"].cpu())
    torch.set_rng_state(state["rng_torch"].cpu())
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all([r.cpu() for r in state["rng_cuda"]])


def validate():
    info = json.loads((FOLDER / "metadata.json").read_text(encoding="utf8"))
    config = yaml.safe_load((FOLDER / "config.yaml").read_text(encoding="utf8"))
    if info["status"] != "interrupted" or config["experiment_id"] != "EXP_006":
        raise ValueError("Only interrupted EXP_006 is supported")
    if (ROOT / ".tools/active_experiment.lock").exists():
        raise RuntimeError("An active lock exists")
    if info["source_fingerprint"] != source_fingerprint():
        raise RuntimeError("Core source fingerprint changed")
    if info["config_sha256"] != sha256(FOLDER / "config.yaml"):
        raise RuntimeError("Config hash changed")
    if info["teacher_sha256"] != sha256(teacher_path()):
        raise RuntimeError("Teacher changed")
    if config != yaml.safe_load((ROOT / "configs/multiseed/EXP_006_kd_seed3407.yaml").read_text()):
        raise RuntimeError("Current and saved configurations differ")
    if config["training"]["num_workers"] != 2:
        raise RuntimeError("Legacy resume is validated for two nonpersistent workers only")
    with (FOLDER / "metrics.csv").open(newline="", encoding="utf8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    with (ROOT / "experiments/registry.csv").open(newline="", encoding="utf8") as stream:
        entries = [r for r in csv.DictReader(stream) if r["experiment_id"] == "EXP_006"]
    if len(entries) != 1 or entries[0]["status"] != "interrupted":
        raise RuntimeError("Registry status mismatch")
    latest = CHECKPOINTS / "EXP_006_latest.pth"
    expected = info.get("latest_checkpoint_sha256")
    if expected is None:
        raise RuntimeError("Recorded latest checkpoint hash is required")
    if sha256(latest) != expected or sha256(ROOT / info["best_checkpoint"]) != info["best_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    state = torch.load(latest, map_location="cpu", weights_only=True)
    epoch = state["epoch"]
    if not (0 < epoch < config["training"]["epochs"]):
        raise ValueError("Checkpoint epoch cannot be resumed")
    if [int(r["epoch"]) for r in rows] != list(range(1, epoch + 1)):
        raise RuntimeError("Metric sequence does not match checkpoint")
    if not all(math.isfinite(float(v)) for r in rows for v in r.values()):
        raise RuntimeError("Nonfinite metrics")
    if (info["last_epoch"] != epoch or state["config"] != config
            or state["git_commit"] != info["git_commit"]
            or state["best_accuracy"] != info["best_accuracy"]):
        raise RuntimeError("Checkpoint provenance mismatch")
    if float(rows[-1]["test_accuracy"]) != info["last_metrics"]["test_accuracy"]:
        raise RuntimeError("Final metric mismatch")
    if max(float(r["test_accuracy"]) for r in rows) != info["best_accuracy"]:
        raise RuntimeError("Best metric mismatch")
    for key in ("rng_torch", "rng_cuda", "loader_rng", "optimizer"):
        if key not in state:
            raise RuntimeError("Missing state: " + key)
    ensure_gates(config)  # Original Teacher, smoke and Baseline gates remain enforced.
    return info, config, state, fields


def summary(info):
    (FOLDER / "summary.md").write_text(
        f'# EXP_006：KD 续训记录\n\n状态：{info["status"]}。已完成 {info["last_epoch"]}/240 epochs。\n\n'
        f'最佳 Top-1：{info["best_accuracy"]}%（epoch {info["best_epoch"]}）。'
        f'最近完成 epoch 的 Top-1：{info["last_metrics"]["test_accuracy"]}%。\n\n'
        '从 epoch 60 checkpoint 恢复，前 60 个 epoch 指标保留；续训来源见 metadata.json 的 resume_sessions。\n'
        '未完成 240 epochs 时均为阶段结果。完成后检查完整结果与异常，再推进 LS-KD。\n',
        encoding="utf8")


def report_incident(info, problem):
    directory = ROOT / "reports/incidents"
    index = max([int(p.name.split("_")[1]) for p in directory.glob("INC_*.md")] + [0]) + 1
    path = directory / f"INC_{index:03d}_EXP_006_resume.md"
    with path.open("x", encoding="utf8") as stream:
        stream.write(f"# INC_{index:03d}：EXP_006 续训异常待检查\n\n"
                     f"状态：OPEN。日期：{now()}。\n\n实际：{problem}\n\n"
                     "预期：指标有限、训练完成；KD 相对 Baseline 的提升为待检验假设。\n\n"
                     "已保留指标、checkpoint、metadata.json、train.log 和 error.log。暂停下一阶段，"
                     "先检查配置、数据、训练日志及结果。不得丢弃无提升结果。\n")
    info["incident"] = path.relative_to(ROOT).as_posix()
    print("INCIDENT: " + info["incident"] + ": " + problem, flush=True)


def run():
    info, config, state, fields = validate()
    check_code_clean()
    lock_path = ROOT / ".tools/active_experiment.lock"
    with lock_path.open("x", encoding="utf8") as stream:
        json.dump({"pid": os.getpid(), "experiment_id": "EXP_006",
                   "started": now(), "mode": "resume"}, stream)
    session = None
    try:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        archive = ROOT / "reports/progress" / ("EXP_006_resume_" + stamp)
        archive.mkdir(exist_ok=False)
        shutil.copytree(FOLDER, archive / "original_experiment")
        shutil.copy2(ROOT / "experiments/registry.csv", archive / "registry_before.csv")
        backup = CHECKPOINTS / ("before_resume_" + stamp)
        backup.mkdir(exist_ok=False)
        for name in ("EXP_006_best.pth", "EXP_006_latest.pth"):
            shutil.copy2(CHECKPOINTS / name, backup / name)
        seed_all(config["seed"])
        settings = config["training"]
        device = torch.device("cuda")
        test_loader = make_loader(ROOT / "data/raw", False, settings["batch_size"], workers=1, seed=config["seed"])
        student = build_model(config["student"]).to(device)
        teacher = load_teacher(device)
        loader = make_loader(ROOT / "data/raw", True, settings["batch_size"],
                             workers=settings["num_workers"], seed=config["seed"])
        optimizer = torch.optim.SGD(student.parameters(), lr=settings["learning_rate"],
                                    momentum=settings["momentum"], weight_decay=settings["weight_decay"])
        restore_state(state, student, optimizer, loader, device)
        session = {"resumed_at": now(), "from_epoch": state["epoch"], "pid": os.getpid(),
                   "driver": "operations/resume_exp006.py", "driver_sha256": sha256(__file__),
                   "driver_git_commit": git_commit(),
                   "checkpoint_sha256": sha256(CHECKPOINTS / "EXP_006_latest.pth"),
                   "archive": archive.relative_to(ROOT).as_posix(),
                   "checkpoint_backup": backup.relative_to(ROOT).as_posix(),
                   "previous_end_time": info.get("end_time"),
                   "previous_stop_reason": info.get("stop_reason"),
                   "state_scope": "model,optimizer,torch_rng,cuda_rng,train_loader_rng",
                   "legacy_rng_note": "Main Python/NumPy states absent; training transforms run in freshly seeded workers. Evaluation transforms deterministic."}
        info.setdefault("resume_sessions", []).append(session)
        info.update(status="running", end_time=None, pid=os.getpid(), resume_supported=True,
                    notes="用户授权从 epoch 60 续训；前段记录已归档，核心源码未变。")
        info.pop("stop_reason", None)
        write_json(FOLDER / "metadata.json", info)
        registry(config, info, info["best_accuracy"])
        summary(info)

        def log(message):
            line = now() + " " + message
            print(line, flush=True)
            with (FOLDER / "train.log").open("a", encoding="utf8") as stream:
                stream.write(line + "\n")

        log("resume_from_epoch=" + str(state["epoch"]) + " driver_sha256=" + session["driver_sha256"])
        best = info["best_accuracy"]
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(state["epoch"] + 1, settings["epochs"] + 1):
            started = time.perf_counter()
            lr = settings["learning_rate"] * settings["lr_decay_rate"] ** sum(
                epoch > stage for stage in settings["lr_decay_stages"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            log(f'epoch={epoch}/{settings["epochs"]} lr={lr}')
            metrics = train_epoch(student, teacher, loader, optimizer, config, device, log)
            metrics.update(evaluate(student, test_loader, device))
            metrics.update(epoch=epoch, learning_rate=lr,
                           epoch_seconds=time.perf_counter() - started,
                           peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
            with (FOLDER / "metrics.csv").open("a", newline="", encoding="utf8") as stream:
                csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore").writerow(metrics)
            saved = {"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "config": config, "git_commit": info["git_commit"],
                     "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                     "loader_rng": loader.generator.get_state(),
                     "resume_driver_sha256": session["driver_sha256"],
                     "resume_driver_git_commit": session["driver_git_commit"]}
            if metrics["test_accuracy"] > best:
                best = metrics["test_accuracy"]
                best_path = CHECKPOINTS / "EXP_006_best.pth"
                atomic_checkpoint(best_path, saved)
                info.update(best_epoch=epoch, best_checkpoint_sha256=sha256(best_path))
            saved["best_accuracy"] = best
            atomic_checkpoint(CHECKPOINTS / "EXP_006_latest.pth", saved)
            info.update(last_epoch=epoch, best_accuracy=best, last_metrics=metrics, updated_at=now(),
                        latest_checkpoint_sha256=sha256(CHECKPOINTS / "EXP_006_latest.pth"))
            write_json(FOLDER / "metadata.json", info)
            registry(config, info, best)
            log(f'epoch_complete={epoch} test_accuracy={metrics["test_accuracy"]:.3f} best={best:.3f} epoch_seconds={metrics["epoch_seconds"]:.1f}')
        baseline = json.loads((ROOT / "experiments/baseline/EXP_004_baseline_seed3407/metadata.json").read_text())
        info.update(status="completed", end_time=now())
        if best <= baseline["best_accuracy"]:
            info.update(status="warning", notes="KD 未超过 Baseline；保留结果并审查，暂不推进下一阶段。")
            report_incident(info, f"KD={best}，Baseline={baseline['best_accuracy']}")
        session["completed_at"] = now()
        write_json(FOLDER / "metadata.json", info)
        registry(config, info, best)
        summary(info)
    except BaseException:
        details = traceback.format_exc()
        with (FOLDER / "error.log").open("a", encoding="utf8") as stream:
            stream.write("\n" + now() + " resume_error\n" + details)
        if session is not None:
            info.update(status="failed", end_time=now(), notes="续训异常；保留现场，检查 error.log 后再推进。")
            report_incident(info, "续训进程出现异常，详见 error.log")
            write_json(FOLDER / "metadata.json", info)
            registry(config, info, info["best_accuracy"])
            summary(info)
        raise
    finally:
        if lock_path.exists() and json.loads(lock_path.read_text())["pid"] == os.getpid():
            lock_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        info, config, state, fields = validate()
        print(json.dumps({"status": "passed", "from_epoch": state["epoch"],
                          "to_epoch": config["training"]["epochs"],
                          "core_fingerprint": source_fingerprint(),
                          "driver_sha256": sha256(__file__)}))
    else:
        run()
