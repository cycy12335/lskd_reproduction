import csv
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import train_epoch
from training import runtime
from models.architectures import build_model


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.fc = torch.nn.Linear(3, 100)

    def forward(self, images):
        return self.fc(self.bn(images)), {}


def test_training_updates_student_but_preserves_teacher_bn_and_weights():
    torch.set_num_threads(2)
    student = TinyModel()
    teacher = TinyModel().eval().requires_grad_(False)
    before_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
    before_student = student.fc.weight.detach().clone()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.arange(8)), batch_size=4)
    config = {"method": "lskd", "distillation": {"temperature": 2., "standardized": True, "ce_weight": 0.1, "kd_weight": 9.}}
    metrics = train_epoch(student, teacher, loader, optimizer, config, "cpu", log=lambda message: None, check_gradients=True)
    assert metrics["train_samples"] == 8
    assert not torch.equal(before_student, student.fc.weight)
    assert not teacher.training
    assert all(p.grad is None for p in teacher.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before_teacher[key], rtol=0, atol=0)


def test_completed_experiment_id_cannot_be_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    (tmp_path / "experiments").mkdir()
    with (tmp_path / "experiments/registry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "status"])
        writer.writeheader()
        writer.writerow({"experiment_id": "EXP_001", "status": "completed"})
    with pytest.raises(FileExistsError):
        runtime.create_experiment({"experiment_id": "EXP_001", "method": "baseline"})


def test_architecture_checkpoint_round_trip(tmp_path):
    torch.set_num_threads(2)
    model = build_model("resnet8x4").eval()
    sample = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(sample)[0]
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    reloaded = build_model("resnet8x4").eval()
    reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        actual = reloaded(sample)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
