import time
import torch
from losses import total_loss


def train_epoch(student, teacher, loader, optimizer, config, device, log=print, check_gradients=False):
    student.train()
    if teacher is not None:
        teacher.eval()
    loss_sum = ce_sum = kd_sum = correct = count = 0
    started = time.perf_counter()
    for step, (images, target) in enumerate(loader, 1):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = student(images)
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits, _ = teacher(images)
        loss, ce, kd = total_loss(logits, teacher_logits, target, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        if check_gradients:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
                raise FloatingPointError("Nonfinite student gradients")
        optimizer.step()
        n = len(target)
        count += n
        loss_sum += loss.item() * n
        ce_sum += ce.item() * n
        kd_sum += kd.item() * n
        correct += logits.argmax(1).eq(target).sum().item()
        if step == 1 or step % 100 == 0 or step == len(loader):
            log(f'step={step}/{len(loader)} loss={loss_sum/count:.5f} accuracy={100*correct/count:.3f} elapsed_s={time.perf_counter()-started:.1f}')
    if count == 0:
        raise ValueError("Empty training dataset")
    return {"train_loss": loss_sum / count, "train_accuracy": 100 * correct / count,
            "train_ce": ce_sum / count, "train_kd": kd_sum / count, "train_samples": count}
