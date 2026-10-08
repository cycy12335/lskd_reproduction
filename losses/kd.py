"""Match official KD.py, including sample standard deviation (correction=1)."""
import torch.nn.functional as F


def standardize(logits):
    return (logits - logits.mean(dim=-1, keepdim=True)) / (
        logits.std(dim=-1, keepdim=True, correction=1) + 1e-7
    )


def kd_loss(student, teacher, temperature, standardized=False):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if standardized:
        student, teacher = standardize(student), standardize(teacher)
    return F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher.detach() / temperature, dim=1),
        reduction="none",
    ).sum(1).mean() * temperature ** 2


def total_loss(student, teacher, target, config):
    ce = F.cross_entropy(student, target)
    if config["method"] == "baseline":
        return ce, ce, ce.new_zeros(())
    settings = config["distillation"]
    kd = kd_loss(student, teacher, settings["temperature"], settings["standardized"])
    return settings["ce_weight"] * ce + settings["kd_weight"] * kd, ce, kd
