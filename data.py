from randaugment import RandAugment
from torchvision import transforms
from torchvision.datasets import CIFAR10

str_augment = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.RandomHorizontalFlip(),
    RandAugment(),
    transforms.ToTensor()
])
test_augment = transforms.Compose([
    transforms.CenterCrop(28),
    transforms.ToTensor()
])
# # taken from https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
# mean = (0.4914, 0.4822, 0.4465)
# std = (0.2023, 0.1994, 0.2010)
# normalize = transforms.Normalize(mean=mean, std=std)
# str_augment = transforms.Compose([
#     transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     normalize,
# ])
# test_augment = transforms.Compose([
#     transforms.CenterCrop(28),
#     transforms.ToTensor(),
#     normalize,
# ])


class TransformTwice:

    def __init__(self, transform) -> None:
        super().__init__()
        self._transform = transform

    def __call__(self, *args, **kwargs):
        return [self._transform(*args, **kwargs), self._transform(*args, **kwargs)]


tra_set = CIFAR10(root="./data", train=True, download=True, transform=TransformTwice(str_augment))
test_set = CIFAR10(root="./data", train=False, download=True, transform=test_augment)
