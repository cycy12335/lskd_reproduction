"""Environment tests only; full distillation pipeline tests come later."""
import torch
import torchvision
from torchvision.ops import nms


def test_torchvision_compiled_cpu_operator():
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
    scores = torch.tensor([0.9, 0.8])
    assert nms(boxes, scores, 0.5).tolist() == [0]


def test_cpu_convolution_backward():
    # Tiny synthetic tensors only; this is not a CIFAR-100 experiment.
    layer = torch.nn.Conv2d(3, 4, 3)
    layer(torch.ones(2, 3, 8, 8)).square().mean().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
