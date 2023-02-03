import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
from typing import Callable

from mlalgos.deep_learning.alexnet.constants_cifar import DataConsts


def get_data_loaders(
    batch_size: int,
    num_workers: int=0,
    val_fraction: float=0.1,
    train_transform: Callable=None,
    test_transform: Callable=None
) -> tuple[torch.utils.data.DataLoader]:
    """
    Creates dataloaders for the CIFAR10 data for training, validation, and test data.

    Args:
        batch_size (int): Size of each training batch
        num_workers (int): Number of workers in the data loaders. Defaults to 0.
        val_fraction (float): Fraction of training data to use for validation. Defaults to 0.1.
        train_transform (Callable): Transforms to perform on the training data.
        test_transform (Callable): Transforms to perfors on the test data.

    Returns:
        tuple[torch.utils.data.DataLoader]: Dataloaders for training, validation, and test data
    """

    if train_transform is None:
            train_transform = transforms.ToTensor()
    if test_transform is None:
            test_transform = transforms.ToTensor()

    train_set = torchvision.datasets.CIFAR10(root='~/datasets', train=True, transform=train_transform)

    val_set = torchvision.datasets.CIFAR10(root='~/datasets', train=True, transform=test_transform)

    test_set = torchvision.datasets.CIFAR10(root='~/datasets', train=False, transform=test_transform)

    num_val = int(val_fraction * DataConsts.NUM_TRAINING)
    train_indices = torch.arange(0, DataConsts.NUM_TRAINING - num_val)
    val_indices = torch.arange(DataConsts.NUM_TRAINING - num_val, DataConsts.NUM_TRAINING)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last=True)

    return train_loader, val_loader, test_loader