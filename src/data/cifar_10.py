import torch
import torchvision
from src.data.transform_utils import preprocess

def get_dataloader(batch_size, train=True):
    dataset = torchvision.datasets.CIFAR10('./src/data/cache', train=train, download=True, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=8)
    return dataloader