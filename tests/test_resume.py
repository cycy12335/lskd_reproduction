import copy
import io
import pytest
import torch
from datasets.cifar100 import make_loader
from training.runtime import seed_all
from training.trainer import train_epoch
from training.evaluator import evaluate
from operations.resume_kd import restore_state


class SmallImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(4, 100)

    def forward(self, x):
        x = self.drop(self.bn(self.conv(x)).mean((2, 3)))
        return self.fc(x), {}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_epoch_boundary_resume_matches_continuous_training(device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("local CUDA unavailable")
    seed_all(42)
    config = {"method": "kd", "distillation": {
        "temperature": 2., "ce_weight": .1, "kd_weight": 9., "standardized": False}}
    student = SmallImageModel().to(device)
    teacher = SmallImageModel().to(device).eval().requires_grad_(False)
    teacher_weights = copy.deepcopy(teacher.state_dict())
    loader = make_loader(tmp_path, True, 8, workers=2, seed=42, fake=True)
    test_loader = make_loader(tmp_path, False, 8, workers=1, seed=42, fake=True)
    optimizer = torch.optim.SGD(student.parameters(), lr=.05, momentum=.9, weight_decay=.0005)
    quiet = lambda message: None
    train_epoch(student, teacher, loader, optimizer, config, device, quiet)
    evaluate(student, test_loader, device)
    buffer = io.BytesIO()
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "loader_rng": loader.generator.get_state()}, buffer)
    expected_metrics = train_epoch(student, teacher, loader, optimizer, config, device, quiet)
    expected_model = copy.deepcopy(student.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    expected_cuda = torch.cuda.get_rng_state_all() if device == "cuda" else []
    expected_loader = loader.generator.get_state().clone()

    # Reconstruct processes' objects, perturb RNG, then load serialized CPU state.
    seed_all(999)
    resumed = SmallImageModel().to(device)
    resumed_teacher = SmallImageModel().to(device).eval().requires_grad_(False)
    resumed_teacher.load_state_dict(teacher_weights)
    resumed_loader = make_loader(tmp_path, True, 8, workers=2, seed=42, fake=True)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.05, momentum=.9, weight_decay=.0005)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    restore_state(state, resumed, resumed_optimizer, resumed_loader, device)
    actual_metrics = train_epoch(resumed, resumed_teacher, resumed_loader,
                                 resumed_optimizer, config, device, quiet)
    assert actual_metrics == (expected_metrics if device == 'cuda' else pytest.approx(expected_metrics, rel=1e-6, abs=1e-7))
    rtol, atol = (0, 0) if device == 'cuda' else (1e-6, 1e-7)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected_model[key], rtol=rtol, atol=atol)
    for key, values in resumed_optimizer.state_dict()["state"].items():
        for field, value in values.items():
            torch.testing.assert_close(value, expected_optimizer["state"][key][field], rtol=rtol, atol=atol)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(expected_loader, resumed_loader.generator.get_state())
    if device == "cuda":
        assert all(torch.equal(a, b) for a, b in zip(expected_cuda, torch.cuda.get_rng_state_all()))
