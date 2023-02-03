class DataConsts:
    IMAGE_SIZE = 32
    CROP_SIZE = 26
    NUM_CLASSES = 10
    NUM_TRAINING = 50000
    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2112, 0.2086, 0.2121]


class Kernel1:
    NUM_CHANNELS = 96
    KERNEL_SIZE = 5
    STRIDE = 2
    PADDING = 0


class Kernel2:
    NUM_CHANNELS = 256
    KERNEL_SIZE = 3
    STRIDE = 1
    PADDING = 1


class Kernel3:
    NUM_CHANNELS = 384
    KERNEL_SIZE = 4
    STRIDE = 1
    PADDING = 1


class Kernel4:
    NUM_CHANNELS = 384
    KERNEL_SIZE = 4
    STRIDE = 1
    PADDING = 1


class Kernel5:
    NUM_CHANNELS = 256
    KERNEL_SIZE = 4
    STRIDE = 1
    PADDING = 1
    OUTPUT_SIZE = 5


class Pool:
    KERNEL_SIZE = 3
    STRIDE = 1


class NormConsts:
    SIZE = 5
    ALPHA = 5e-4
    BETA = 0.75
    K = 2


class FullyConnected:
    # Make sure that the dimensions work out and adjust accordingly if changed
    INPUT = Kernel5.OUTPUT_SIZE * Kernel5.OUTPUT_SIZE * Kernel5.NUM_CHANNELS
    NUM = 4096


# Sequence of sizes, given input 26:
# 11, 9, 9, 7, 7, 7, 7, 5