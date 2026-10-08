import pytest
import torch
from losses import kd_loss, standardize


def test_standardization_has_zero_mean_and_unit_sample_std():
    values = torch.tensor([[1., 2., 5.], [-10., 0., 8.]], dtype=torch.float64)
    normalized = standardize(values)
    torch.testing.assert_close(normalized.mean(1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(normalized.std(1), torch.ones(2, dtype=torch.float64), atol=1e-7, rtol=0)


def test_lskd_invariant_to_independent_positive_scale_and_shift():
    torch.manual_seed(5)
    student, teacher = torch.randn(3, 100, dtype=torch.float64), torch.randn(3, 100, dtype=torch.float64)
    original = kd_loss(student, teacher, 2, True)
    shifted = kd_loss(3 * student + 17, 7 * teacher - 25, 2, True)
    torch.testing.assert_close(original, shifted, rtol=1e-6, atol=1e-8)
    assert not torch.isclose(kd_loss(student, teacher, 2), kd_loss(3 * student + 17, 7 * teacher - 25, 2))


@pytest.mark.parametrize("temperature", [1., 2., 4., 8.])
def test_kd_matches_probability_definition_and_stops_teacher_gradient(temperature):
    student = torch.tensor([[1., -1., 0.], [0., 2., -1.]], requires_grad=True)
    teacher = torch.tensor([[2., 0., -1.], [1., 1., 1.]], requires_grad=True)
    p = torch.softmax(teacher.detach() / temperature, dim=1)
    q = torch.softmax(student.detach() / temperature, dim=1)
    expected = (p * (p.log() - q.log())).sum(1).mean() * temperature ** 2
    actual = kd_loss(student, teacher, temperature)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()


def test_constant_logits_are_finite():
    constant = torch.ones(2, 100, requires_grad=True)
    loss = kd_loss(constant, torch.ones_like(constant), 2., True)
    loss.backward()
    assert torch.isfinite(constant.grad).all()
