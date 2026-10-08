import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = correct5 = count = 0
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits, _ = model(images)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        n = len(target)
        loss_sum += loss.item() * n
        correct += logits.argmax(1).eq(target).sum().item()
        correct5 += logits.topk(5, dim=1).indices.eq(target[:, None]).any(1).sum().item()
        count += n
    if count == 0:
        raise ValueError("Empty evaluation loader")
    return {"test_loss": loss_sum / count, "test_accuracy": 100 * correct / count,
            "test_top5_accuracy": 100 * correct5 / count, "test_samples": count}
