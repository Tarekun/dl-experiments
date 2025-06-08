import torch
import torch.nn as nn
import torch.nn.functional as F
from train import device


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, conv_layers=2, lin_layers=1, starting_channels=3):
        """Default Convolutional network class for classification.
        Controllable hyper-parameters are:
        * `num_classes`: the number of output classes
        * `conv_layers`: the number of convolutional layers to be used
        * `lin_layers`: the number of fully connected layers following convolutionals
        * `starting_channels`: the number of channels/feature maps the first layer has

        The first convolutional layer will have `starting_channels` input channels and
        64 output channels. All following layers will have double the input channels
        as outputs.\n
        The first linear layer will have an input number of neuron compatible with the
        output of convolutional layers and output 1024 neurons. All following layers will
        have half the input neurons as output and the last one will have `num_classes`
        outputs
        """
        super(SimpleCNN, self).__init__()

        self.conv_stack = SimpleCNN._make_conv_stack(starting_channels, conv_layers)
        starting_neurons = self._compute_flattened_size(starting_channels)
        self.fc_stack = SimpleCNN._make_fc_stack(
            starting_neurons, lin_layers, num_classes
        )

    @staticmethod
    def _make_conv_stack(starting_channels: int, conv_layers: int):
        conv_modules = []
        in_channels = starting_channels
        out_channels = 64

        for _ in range(conv_layers):
            conv_modules.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            conv_modules.append(nn.ReLU())
            conv_modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels
            out_channels *= 2

        return nn.Sequential(*conv_modules)

    @staticmethod
    def _make_fc_stack(starting_neurons: int, lin_layers: int, num_classes: int):
        fc_modules = []
        in_features = starting_neurons
        out_features = 1024
        for _ in range(lin_layers - 1):  # Create all but the last fully connected layer
            fc_modules.append(nn.Linear(in_features, out_features))
            fc_modules.append(nn.ReLU())

            in_features = out_features
            out_features //= 2

        fc_modules.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*fc_modules)

    def _compute_flattened_size(self, starting_channels):
        """Computes the flattened size of the convolutional layers output.
        Computation is performed by a forward pass on self.conv_stack with
        dummy data"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, starting_channels, 32, 32)
            output = self.conv_stack(dummy_input)
            flattened_size = output.view(1, -1).size(1)
        return flattened_size

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


# --- ResNet Implementation for CIFAR-10/100 ---


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample: nn.Module | None = None
    ):
        """Residual CNN block. It includes 2 convolutional layers followed by batch normilization
        with ReLU activation function

        Parameters
        ----------
        downsample:
            Downsampling module to match residual shape with the output of convolutions
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers: list[int], num_classes=10, in_channels=3):
        super(ResNet, self).__init__()

        # first embedding, no pooling too soon
        self.base_channels = 64
        self.conv1 = conv3x3(in_channels, self.base_channels)
        self.bn1 = nn.BatchNorm2d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stack = nn.Sequential(
            *[
                self._make_layer(self.base_channels * (2**i), num_blocks).to(device)
                for i, num_blocks in enumerate(layers)
            ]
        )
        # self.layer1 = self._make_layer(64, layers[0])
        # self.layer2 = self._make_layer(128, layers[1], stride=2)
        # self.layer3 = self._make_layer(256, layers[2], stride=2)
        # self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResidualBlock.expansion, num_classes)

        # initialize weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Module:
        downsample = None
        if stride != 1 or self.base_channels != out_channels * ResidualBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.base_channels,
                    out_channels * ResidualBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * ResidualBlock.expansion),
            )

        layers = []
        layers.append(
            ResidualBlock(self.base_channels, out_channels, stride, downsample)
        )
        self.base_channels = out_channels * ResidualBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.base_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stack(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=10, in_channels=3):
    """Constructs a ResNet-18 model for CIFAR-sized inputs."""
    return ResNet([2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)
