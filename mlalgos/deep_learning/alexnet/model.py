import torch
import torch.nn as nn
import torch.nn.functional as F

from constants_cifar import (
    DataConsts,
    Kernel1,
    Kernel2,
    Kernel3,
    Kernel4,
    Kernel5,
    Pool,
    NormConsts,
    FullyConnected)


class AlexNetModel(nn.Module):
    """
    The model definition described in the AlexNet paper.
    """
    def __init__(self):
        super(AlexNetModel, self).__init__()
        conv1 = nn.Conv2d(in_channels=3,
                      out_channels=Kernel1.NUM_CHANNELS,
                      kernel_size=Kernel1.KERNEL_SIZE,
                      stride=Kernel1.STRIDE,
                      padding=Kernel1.PADDING)

        conv2 = nn.Conv2d(in_channels=Kernel1.NUM_CHANNELS,
                      out_channels=Kernel2.NUM_CHANNELS,
                      kernel_size=Kernel2.KERNEL_SIZE,
                      stride=Kernel2.STRIDE,
                      padding=Kernel2.PADDING)

        conv3 = nn.Conv2d(in_channels=Kernel2.NUM_CHANNELS,
                      out_channels=Kernel3.NUM_CHANNELS,
                      kernel_size=Kernel3.KERNEL_SIZE,
                      stride=Kernel3.STRIDE,
                      padding=Kernel3.PADDING)

        conv4 = nn.Conv2d(in_channels=Kernel3.NUM_CHANNELS,
                      out_channels=Kernel4.NUM_CHANNELS,
                      kernel_size=Kernel4.KERNEL_SIZE,
                      stride=Kernel4.STRIDE,
                      padding=Kernel4.PADDING)
        
        conv5 = nn.Conv2d(in_channels=Kernel4.NUM_CHANNELS,
                      out_channels=Kernel5.NUM_CHANNELS,
                      kernel_size=Kernel5.KERNEL_SIZE,
                      stride=Kernel5.STRIDE,
                      padding=Kernel5.PADDING)
        
        activation = nn.ReLU(inplace=True)

        norm = nn.LocalResponseNorm(size=NormConsts.SIZE,
                                    alpha=NormConsts.ALPHA,
                                    beta=NormConsts.BETA,
                                    k=NormConsts.K)

        pool = nn.MaxPool2d(kernel_size=Pool.KERNEL_SIZE, stride=Pool.STRIDE)

        linear1 = nn.Linear(FullyConnected.INPUT, FullyConnected.NUM)

        linear2 = nn.Linear(FullyConnected.NUM, FullyConnected.NUM)

        linear3 = nn.Linear(FullyConnected.NUM, DataConsts.NUM_CLASSES)

        dropout = nn.Dropout(0.5)

        self.convnet = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            # activation,
            norm,
            nn.MaxPool2d(kernel_size=Pool.KERNEL_SIZE, stride=Pool.STRIDE),
            # pool,

            conv2,
            nn.ReLU(inplace=True),
            # activation,
            norm,
            nn.MaxPool2d(kernel_size=Pool.KERNEL_SIZE, stride=Pool.STRIDE),
            # pool,
            
            conv3,
            nn.ReLU(inplace=True),
            # activation,

            conv4,
            nn.ReLU(inplace=True),
            # activation,

            conv5,
            nn.ReLU(inplace=True),
            # activation,
            nn.MaxPool2d(kernel_size=Pool.KERNEL_SIZE, stride=Pool.STRIDE)
            # pool
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((Kernel5.OUTPUT_SIZE, Kernel5.OUTPUT_SIZE))

        self.fullyconnected = nn.Sequential(
            nn.Dropout(0.5),
            # dropout,
            linear1,
            nn.ReLU(inplace=True),
            # activation,

            nn.Dropout(0.5),
            # dropout,
            linear2,
            nn.ReLU(inplace=True),
            # activation,

            linear3
        )

    def forward(self, x):
        x = self.convnet(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fullyconnected(x)
        # Return the output directly, referred to as logits
        # since that is what cross_entropy takes as argument
        return logits
