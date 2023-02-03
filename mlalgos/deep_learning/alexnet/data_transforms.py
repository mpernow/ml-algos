import torchvision.transforms as transforms

from mlalgos.deep_learning.alexnet.constants_cifar import DataConsts
from mlalgos.utils.data_utils import TransformPCA


normalise = transforms.Normalize(mean=DataConsts.MEAN, std=DataConsts.STD)


cifar_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalise,
     transforms.RandomCrop((DataConsts.CROP_SIZE, DataConsts.CROP_SIZE)),
     transforms.Resize((DataConsts.IMAGE_SIZE, DataConsts.IMAGE_SIZE)),
     transforms.RandomHorizontalFlip(),
     TransformPCA()
    ])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     normalise,
     transforms.Resize((DataConsts.IMAGE_SIZE, DataConsts.IMAGE_SIZE))]
)
