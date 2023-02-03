import torch
import torchvision
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

from constants_cifar import DataConsts


def get_data_loaders(
    batch_size,
    num_workers=0,
    val_fraction=0.1,
    train_transform=None,
    test_transform=None
):

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