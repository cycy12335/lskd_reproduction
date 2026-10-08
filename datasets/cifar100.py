import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def transform(train):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_loader(root, train, batch_size, workers=2, seed=42, tiny=False, fake=False):
    if fake:
        dataset = datasets.FakeData(2 * batch_size, (3, 32, 32), 100, transform=transform(train))
    else:
        dataset = datasets.CIFAR100(root, train=train, transform=transform(train), download=False)
        if tiny:
            dataset = Subset(dataset, list(range(2 * batch_size)))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
