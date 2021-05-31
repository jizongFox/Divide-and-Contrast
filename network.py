from collections import OrderedDict
from contextlib import contextmanager

import torch
from loguru import logger
from torch import nn
from torchvision.models import resnet50


class Hook:

    def __init__(self) -> None:
        super().__init__()
        self.activation = None

    def __call__(self, model, input, output):
        self.activation = input[0]

    def get_activation(self):
        return self.activation


class Model(nn.Module):
    def __init__(self, input_dim, num_classes, pretrained=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self._resnet = resnet50(pretrained=pretrained)
        if input_dim != 3:
            self._resnet.conv1 = nn.Conv2d(self.input_dim, self.inplanes, kernel_size=7, stride=2, padding=3,  # noqa
                                           bias=False)
        if num_classes != 1000:
            fc_input_dim = self._resnet.fc.in_features
            self._resnet.fc = nn.Linear(fc_input_dim, num_classes)

        self._hook = Hook()
        self._hook_handler = self._resnet.fc.register_forward_hook(self._hook)

    def forward(self, input_):
        return self._resnet(input_), self._hook.get_activation()

    @contextmanager
    def set_grad(self, enable_extractor=True, enable_fc=True):
        logger.debug(f"setting grad: {enable_extractor} for extractor, grad: {enable_fc} for fc")
        previous_state = OrderedDict()
        for name, param in self.named_parameters():
            previous_state[name] = param.requires_grad
            param.requires_grad = enable_fc if "fc" in name else enable_extractor
        yield self
        logger.debug(f"restore grad")
        for name, param in self.named_parameters():
            param.requires_grad = previous_state[name]

    @property
    def feature_dim(self):
        return self._resnet.fc.in_features


class Projector(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self._projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_features=output_dim)
        )

    def forward(self, input_):
        return self._projector(input_)

    @contextmanager
    def set_grad(self, enable=True):
        previous_state = OrderedDict()
        for name, param in self.named_parameters():
            previous_state[name] = param.requires_grad
            param.requires_grad = enable
        yield self
        for name, param in self.named_parameters():
            param.requires_grad = previous_state[name]


def detach_grad(model):
    for param in model.parameters():
        param.detach_()
    return model


if __name__ == '__main__':
    image = torch.randn(10, 3, 225, 225)
    model = Model(3, 10, pretrained=True)
    prediction, features = model(image)
    print(prediction.shape, features.shape)
