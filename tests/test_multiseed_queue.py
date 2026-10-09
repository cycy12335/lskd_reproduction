import json
import pytest
import yaml
from operations.multiseed_queue import audit


@pytest.mark.parametrize("status", ["warning", "failed", "interrupted", "running"])
def test_audit_blocks_noncompleted_experiment(tmp_path, status):
    (tmp_path / "metadata.json").write_text(json.dumps({"status": status, "experiment_id": "EXP_004"}))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"training": {"epochs": 2}}))
    with pytest.raises(RuntimeError, match="do not advance"):
        audit(tmp_path)


def test_audit_blocks_missing_epoch(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "completed", "last_epoch": 2}))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"training": {"epochs": 2}}))
    (tmp_path / "metrics.csv").write_text("epoch,test_accuracy\n1,72.0\n")
    with pytest.raises(RuntimeError, match="Incomplete epoch"):
        audit(tmp_path)


def test_audit_blocks_nonfinite_metric(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "completed", "last_epoch": 1}))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"training": {"epochs": 1}}))
    (tmp_path / "metrics.csv").write_text("epoch,test_accuracy\n1,nan\n")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        audit(tmp_path)
