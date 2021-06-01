from torchvision import transforms
from torchvision.datasets import CIFAR10

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
normalize = transforms.Normalize(mean=mean, std=std)
pretrain_augment = transforms.Compose([
    transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


class TransformTwice:

    def __init__(self, transform) -> None:
        super().__init__()
        self._transform = transform

    def __call__(self, *args, **kwargs):
        return [self._transform(*args, **kwargs), self._transform(*args, **kwargs)]


def get_pretrain_dataset():
    tra_set = CIFAR10(root="./data", train=True, download=True, transform=TransformTwice(pretrain_augment))
    return tra_set


def get_train_datasets():
    tra_set = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    return tra_set, test_set
