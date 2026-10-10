"""Serial local multi-seed orchestration. Does not modify the training pipeline."""
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
import re
import statistics
import subprocess
import traceback
import shutil
import time
import torch
import yaml
from models.architectures import build_model
from training.runtime import (sha256, source_fingerprint, write_json, now,
                              check_code_clean, git_commit)
from training.runner import ensure_gates

PLAN = ROOT / "configs/multiseed/queue.json"
DIRECTORY = ROOT / "reports/progress/multiseed_20261009"
LOCK = ROOT / ".tools/multiseed_queue.lock"
STOP = ROOT / ".tools/multiseed_stop_requested.json"
STATE = DIRECTORY / "state.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def audit(folder):
    folder = Path(folder)
    m = read_json(folder / "metadata.json")
    c = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf8"))
    if m["status"] != "completed":
        raise RuntimeError(f'{m["experiment_id"]} status={m["status"]}; do not advance')
    with (folder / "metrics.csv").open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    n = c["training"]["epochs"]
    if [int(r["epoch"]) for r in rows] != list(range(1, n + 1)) or m["last_epoch"] != n:
        raise RuntimeError("Incomplete epoch sequence")
    if not all(math.isfinite(float(v)) for r in rows for v in r.values()):
        raise RuntimeError("Nonfinite metrics")
    if list(map(int, re.findall(r"epoch_complete=(\d+)", (folder / "train.log").read_text(encoding="utf8")))) != list(range(1, n + 1)):
        raise RuntimeError("Incomplete completion log")
    if (folder / "error.log").stat().st_size:
        raise RuntimeError("Nonempty experiment error log")
    best = max(rows, key=lambda r: float(r["test_accuracy"]))
    if float(best["test_accuracy"]) != m["best_accuracy"] or int(best["epoch"]) != m["best_epoch"]:
        raise RuntimeError("Best accuracy mismatch")
    if any(float(rows[-1][k]) != float(v) for k, v in m["last_metrics"].items() if k in rows[-1]):
        raise RuntimeError("Final metric mismatch")
    for r in rows:
        settings = c["training"]
        lr = settings["learning_rate"] * settings["lr_decay_rate"] ** sum(int(r["epoch"]) > s for s in settings["lr_decay_stages"])
        if not math.isclose(float(r["learning_rate"]), lr):
            raise RuntimeError("Learning rate mismatch")
    if sha256(folder / "config.yaml") != m["config_sha256"] or source_fingerprint() != m["source_fingerprint"]:
        raise RuntimeError("Configuration/source hash mismatch")
    checkpoints = {}
    for name, epoch in (("best", m["best_epoch"]), ("latest", n)):
        p = ROOT / "models/checkpoints" / c["method"] / c["experiment_id"] / (c["experiment_id"] + "_" + name + ".pth")
        digest = sha256(p)
        if name == "best" and digest != m["best_checkpoint_sha256"]:
            raise RuntimeError("Best checkpoint digest mismatch")
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if ck["epoch"] != epoch or ck["config"] != c or ck["git_commit"] != m["git_commit"]:
            raise RuntimeError("Checkpoint provenance mismatch")
        model = build_model(c["student"])
        model.load_state_dict(ck["model"], strict=True)
        if not all(torch.isfinite(t).all() for t in ck["model"].values()):
            raise RuntimeError("Nonfinite checkpoint")
        checkpoints[name] = {"sha256": digest, "epoch": epoch, "path": p.relative_to(ROOT).as_posix()}
    return {"checked_at": now(), "experiment_id": c["experiment_id"], "method": c["method"],
            "seed": c["seed"], "status": "passed", "best_accuracy": m["best_accuracy"],
            "final_accuracy": m["last_metrics"]["test_accuracy"], "checkpoints": checkpoints}


def preflight():
    if os.environ.get("GITHUB_ACTIONS"):
        raise RuntimeError("Local GPU only")
    if LOCK.exists() or (ROOT / ".tools/active_experiment.lock").exists() or STOP.exists():
        raise RuntimeError("Existing lock or stop request; inspect before starting")
    if STATE.exists():
        raise RuntimeError("Queue already used; do not restart blindly")
    jobs = read_json(PLAN)
    expected = [("EXP_004", "baseline", 3407), ("EXP_006", "kd", 3407), ("EXP_008", "lskd", 3407),
                ("EXP_005", "baseline", 2024), ("EXP_007", "kd", 2024), ("EXP_009", "lskd", 2024)]
    if [(j["experiment_id"], j["method"], j["seed"]) for j in jobs] != expected:
        raise RuntimeError("Unexpected experiment schedule")
    with (ROOT / "experiments/registry.csv").open(newline="", encoding="utf8") as stream:
        used = {r["experiment_id"] for r in csv.DictReader(stream)}
    base_configs = {"baseline": "configs/baseline/resnet8x4_cifar100.yaml",
                    "kd": "configs/kd/resnet32x4_resnet8x4.yaml",
                    "lskd": "configs/lskd/resnet32x4_resnet8x4.yaml"}
    for j in jobs:
        c = yaml.safe_load((ROOT / j["config"]).read_text())
        template = yaml.safe_load((ROOT / base_configs[j["method"]]).read_text())
        template.update(seed=j["seed"], experiment_id=j["experiment_id"])
        if c != template:
            raise RuntimeError("Only seed and experiment ID may change")
        eid = j["experiment_id"]
        if eid in used or list((ROOT / "experiments").glob(f"*/{eid}_*")) or (ROOT / "models/checkpoints" / j["method"] / eid).exists():
            raise RuntimeError("Experiment ID or directory already used: " + eid)
    for method, eid in (("baseline", "EXP_001"), ("kd", "EXP_002"), ("lskd", "EXP_003")):
        audit(ROOT / "experiments" / method / f"{eid}_{method}_seed42")
    ensure_gates(yaml.safe_load((ROOT / jobs[0]["config"]).read_text()))
    return jobs


def chinese_summary(folder):
    m = read_json(folder / "metadata.json")
    text = (f'# {m["experiment_id"]}：多种子实验总结\n\n'
            f'状态：{m["status"]}；seed：{m["seed"]}。\n\n'
            f'完成 epoch：{m.get("last_epoch", 0)}；最佳 Top-1：{m.get("best_accuracy", "尚无")}%（epoch {m.get("best_epoch", "尚无")}）。\n'
            f'最近完成 epoch 的 Top-1：{m.get("last_metrics", {}).get("test_accuracy", "尚无")}%。\n\n'
            'completed 且通过队列完整性检查时才是最终结果；其他状态须检查日志和异常记录，不能推进。\n'
            '原始指标、源码提交和 checkpoint 溯源见同目录 metadata.json、metrics.csv；队列验收见 reports/progress/multiseed_20261009。\n')
    (folder / "summary.md").write_text(text, encoding="utf8")
    if m.get("incident"):
        p = ROOT / m["incident"]
        text = p.read_text(encoding="utf8")
        for old, new in (("Experiment:", "关联实验："), ("Date:", "日期："), ("Git commit:", "代码提交："),
                         ("Status:", "状态："), ("Severity:", "严重程度："), ("Expected:", "预期："),
                         ("Actual:", "实际："), ("Configuration:", "配置："), ("Error log:", "错误日志："),
                         ("Initial diagnosis: inspect saved logs, configuration, dataset and checkpoint before advancing.",
                          "初步处理：先检查保存的日志、配置、数据及 checkpoint，再决定是否推进。")):
            text = text.replace(old, new)
        p.write_text(text, encoding="utf8")


def aggregate():
    results = []
    for method in ("baseline", "kd", "lskd"):
        values = {}
        for p in (ROOT / "experiments" / method).glob("*/metadata.json"):
            m = read_json(p)
            if m["seed"] in (42, 3407, 2024) and m["status"] == "completed":
                if m["seed"] in values:
                    raise RuntimeError("Ambiguous seed result")
                values[m["seed"]] = m
        if set(values) != {42, 3407, 2024}:
            raise RuntimeError("Incomplete three-seed results")
        best = [values[s]["best_accuracy"] for s in (42, 3407, 2024)]
        final = [values[s]["last_metrics"]["test_accuracy"] for s in (42, 3407, 2024)]
        results.append({"method": method, "seed42_best": best[0], "seed3407_best": best[1],
                        "seed2024_best": best[2], "best_mean": statistics.mean(best),
                        "best_sample_sd": statistics.stdev(best), "final_mean": statistics.mean(final),
                        "final_sample_sd": statistics.stdev(final)})
    p = ROOT / "results/statistics/multi_seed_results.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# 多种子阶段汇总", "", "样本标准差使用 n−1 分母；三个种子为 42、3407、2024。", "",
             "| 方法 | 最佳 Top-1 均值 ± SD | 最终 Top-1 均值 ± SD |",
             "|---|---:|---:|"]
    for r in results:
        lines.append(f'| {r["method"]} | {r["best_mean"]:.2f} ± {r["best_sample_sd"]:.2f} | {r["final_mean"]:.2f} ± {r["final_sample_sd"]:.2f} |')
    lines += ["", "此为三个种子的描述统计，不能单凭均值排序宣称统计显著；后续需要配对差值及温度消融。"]
    (DIRECTORY / "汇总.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def preflight_resume():
    if os.environ.get("GITHUB_ACTIONS") or LOCK.exists() or (ROOT / ".tools/active_experiment.lock").exists():
        raise RuntimeError("Local idle queue required")
    previous = read_json(STATE)
    jobs = read_json(PLAN)
    if previous["status"] != "paused" or previous["current"] != "EXP_006":
        raise RuntimeError("Only the user-paused EXP_006 queue is supported")
    if previous["jobs"] != jobs or previous["plan_sha256"] != sha256(PLAN):
        raise RuntimeError("Queue plan changed")
    if previous["core_fingerprint"] != source_fingerprint():
        raise RuntimeError("Core source changed")
    if [r["experiment_id"] for r in previous["completed"]] != ["EXP_004"]:
        raise RuntimeError("Unexpected completed prefix")
    completed = audit(ROOT / "experiments/baseline/EXP_004_baseline_seed3407")
    if completed["checkpoints"] != previous["completed"][0]["checkpoints"]:
        raise RuntimeError("Completed experiment changed")
    from operations.resume_exp006 import validate
    validate()
    with (ROOT / "experiments/registry.csv").open(newline="", encoding="utf8") as stream:
        used = {r["experiment_id"] for r in csv.DictReader(stream)}
    for j in jobs[2:]:
        eid = j["experiment_id"]
        if eid in used or list((ROOT / "experiments").glob(f"*/{eid}_*")) or (ROOT / "models/checkpoints" / j["method"] / eid).exists():
            raise RuntimeError("Pending experiment already used: " + eid)
        base = "configs/baseline/resnet8x4_cifar100.yaml" if j["method"] == "baseline" else f'configs/{j["method"]}/resnet32x4_resnet8x4.yaml'
        expected = yaml.safe_load((ROOT / base).read_text())
        expected.update(seed=j["seed"], experiment_id=eid)
        if yaml.safe_load((ROOT / j["config"]).read_text()) != expected:
            raise RuntimeError("Pending configuration changed")
    return jobs[1:]


def run(resume=False):
    jobs = preflight_resume() if resume else preflight()
    check_code_clean()
    with LOCK.open("x", encoding="utf8") as stream:
        json.dump({"pid": os.getpid(), "started_at": now(), "plan": PLAN.relative_to(ROOT).as_posix()}, stream)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "pid": os.getpid(), "started_at": now(), "git_commit": git_commit(),
             "controller_sha256": sha256(__file__), "plan_sha256": sha256(PLAN),
             "core_fingerprint": source_fingerprint(), "jobs": jobs, "completed": []}
    if resume:
        prior = read_json(STATE)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        shutil.copy2(STATE, DIRECTORY / ("before_resume_" + stamp + ".json"))
        if STOP.exists():
            shutil.copy2(STOP, DIRECTORY / ("stop_request_before_resume_" + stamp + ".json"))
            STOP.unlink()
        state["jobs"] = prior["jobs"]
        state["completed"] = prior["completed"]
        state["previous_session"] = {k: prior[k] for k in ("pid", "started_at", "controller_sha256", "git_commit")}
        state["resumed_at"] = now()
        state["resume_driver_sha256"] = sha256(ROOT / "operations/resume_exp006.py")
    try:
        for j in jobs:
            if STOP.exists():
                state.update(status="stopped", reason="用户停止请求；未启动下一实验")
                break
            if sha256(__file__) != state["controller_sha256"] or sha256(PLAN) != state["plan_sha256"]:
                raise RuntimeError("Queue controller or plan changed during execution")
            c = yaml.safe_load((ROOT / j["config"]).read_text())
            ensure_gates(c)
            eid = j["experiment_id"]
            tag = "_resume_" + stamp if resume else ""
            stdout = ROOT / "logs/system" / f"multiseed_{eid}{tag}.stdout.log"
            stderr = ROOT / "logs/system" / f"multiseed_{eid}{tag}.stderr.log"
            state.update(current=eid, current_config_sha256=sha256(ROOT / j["config"]))
            write_json(STATE, state)
            env = dict(os.environ, PYTHONUTF8="1")
            with stdout.open("x", encoding="utf8") as out, stderr.open("x", encoding="utf8") as err:
                command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", f'scripts/train_{j["method"]}.py', "--config", j["config"]]
                if resume and eid == "EXP_006":
                    if sha256(ROOT / "operations/resume_exp006.py") != state["resume_driver_sha256"]:
                        raise RuntimeError("Resume driver changed")
                    command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", "operations/resume_exp006.py"]
                child = subprocess.Popen(command,
                                         cwd=ROOT, stdout=out, stderr=err, env=env)
                state.update(child_pid=child.pid, stdout=str(stdout), stderr=str(stderr))
                write_json(STATE, state)
                write_json(ROOT / ".tools/last_launch.json",
                           {"pid": child.pid, "controller_pid": os.getpid(), "method": j["method"],
                            "experiment_id": eid, "started_at": now(), "stdout": str(stdout), "stderr": str(stderr)})
                code = child.wait()
            folder = ROOT / "experiments" / j["method"] / f'{eid}_{j["method"]}_seed{j["seed"]}'
            if (folder / "metadata.json").exists():
                chinese_summary(folder)
            if code != 0:
                raise RuntimeError(f"{eid} child exit code={code}")
            result = audit(folder)
            if stderr.stat().st_size:
                raise RuntimeError(f"{eid} has stderr output; review before advancing")
            write_json(DIRECTORY / f"{eid}_audit.json", result)
            state["completed"].append(result)
            write_json(STATE, state)
        else:
            aggregate()
            state.update(status="completed", finished_at=now())
        write_json(STATE, state)
    except BaseException:
        state.update(status="halted", halted_at=now(), error=traceback.format_exc())
        write_json(STATE, state)
        (DIRECTORY / "队列停止说明.md").write_text(
            "# 多种子队列已停止\n\n未继续后续实验。请先检查 state.json、当前实验 error.log 和 reports/incidents。\n\n"
            "原始错误信息：\n\n```text\n" + state["error"] + "\n```\n", encoding="utf8")
        raise
    finally:
        if LOCK.exists() and read_json(LOCK)["pid"] == os.getpid():
            LOCK.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps({"status": "passed", "jobs": preflight_resume() if args.resume else preflight(), "core_fingerprint": source_fingerprint()}))
    else:
        run(resume=args.resume)
