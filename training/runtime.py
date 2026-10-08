import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "eb569ac3783f2eed16fbc2b021101175af4fb2f2"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def source_fingerprint():
    h = hashlib.sha256()
    for directory in ("models/architectures", "losses", "datasets", "training", "scripts"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            h.update(path.relative_to(ROOT).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def check_code_clean():
    output = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
    unexpected = [line for line in output.splitlines() if not line[3:].replace("\\", "/").startswith(("experiments/", "reports/", "results/"))]
    if unexpected:
        raise RuntimeError("Commit source/config changes before experiments: " + repr(unexpected))
    return output


def metadata(config):
    return {
        "experiment_id": config["experiment_id"], "start_time": now(), "end_time": None, "status": "running",
        "git_commit": git_commit(), "git_working_tree": check_code_clean(),
        "source_fingerprint": source_fingerprint(), "upstream_commit": UPSTREAM_COMMIT,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "seed": config.get("seed"), "precision": "FP32", "amp": False,
        "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False,
    }


def teacher_path():
    return ROOT / "models/pretrained/teacher/teacher_resnet32x4_cifar100.pth"


def load_teacher(device):
    from models.architectures import build_model
    model = build_model("resnet32x4")
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(teacher_path(), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def registry(config, info, accuracy=""):
    path = ROOT / "experiments/registry.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    eid = config["experiment_id"]
    row = dict.fromkeys(fields, "")
    row.update(experiment_id=eid, date=info["start_time"][:10], method=config["method"],
               teacher=config.get("teacher", ""), student=config.get("student", ""),
               temperature=config.get("distillation", {}).get("temperature", ""),
               seed=config.get("seed", ""), status=info["status"], best_accuracy=accuracy,
               git_commit=info["git_commit"], notes=info.get("notes", ""))
    rows = [r for r in rows if r["experiment_id"] != eid] + [row]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def create_experiment(config):
    eid = config["experiment_id"]
    if not (len(eid) == 7 and eid.startswith("EXP_") and eid[4:].isdigit()):
        raise ValueError("Experiment ID must be EXP_XXX")
    with (ROOT / "experiments/registry.csv").open(newline="", encoding="utf-8") as f:
        if any(row["experiment_id"] == eid for row in csv.DictReader(f)):
            raise FileExistsError(f"Experiment ID already reserved: {eid}")
    if list((ROOT / "experiments").glob(f"*/{eid}_*")):
        raise FileExistsError(f"Experiment directory already exists: {eid}")
    info = metadata(config)
    folder = ROOT / "experiments" / config["method"] / f'{eid}_{config["method"]}_seed{config.get("seed", "none")}'
    folder.mkdir(parents=True, exist_ok=False)
    import yaml
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (folder / "error.log").touch()
    write_json(folder / "metadata.json", info)
    registry(config, info)
    return folder, info


def incident(folder, config, info, problem, expected):
    directory = ROOT / "reports/incidents"
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob("INC_*.md"))
    index = max([int(p.name.split("_")[1]) for p in existing] + [0]) + 1
    path = directory / f'INC_{index:03d}_{config["experiment_id"]}.md'
    path.write_text(f'# INC_{index:03d}\n\nExperiment: {config["experiment_id"]}\n\nDate: {now()}\n\n'
                    f'Git commit: {info["git_commit"]}\n\nStatus: OPEN\n\nSeverity: High\n\n'
                    f'Expected: {expected}\n\nActual: {problem}\n\n'
                    f'Configuration: {folder.relative_to(ROOT).as_posix()}/config.yaml\n\n'
                    f'Error log: {folder.relative_to(ROOT).as_posix()}/error.log\n\n'
                    'Initial diagnosis: inspect saved logs, configuration, dataset and checkpoint before advancing.\n', encoding="utf-8")
    info["incident"] = path.relative_to(ROOT).as_posix()
    print(f'INCIDENT: {info["incident"]}: {problem}', flush=True)
