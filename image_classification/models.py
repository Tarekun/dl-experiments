import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
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

    def _make_fc_stack(starting_neurons: int, lin_layers: int, num_classes: int):
        fc_modules = []
        in_features = starting_neurons
        out_features = 1024
        for _ in range(lin_layers - 1):  # Create all but the last fully connected layer
            fc_modules.append(nn.Linear(in_features, out_features))
            fc_modules.append(
                nn.ReLU()
            )  # ReLU activation after each fully connected layer
            in_features = out_features
            out_features //= (
                2  # Halve the number of neurons for each fully connected layer
            )
        fc_modules.append(nn.Linear(in_features, num_classes))

        return nn.Sequential(*fc_modules)

    def __init__(self, num_classes, conv_layers=2, lin_layers=1, starting_channels=3):
        super(SimpleCNN, self).__init__()

        self.conv_stack = SimpleCNN._make_conv_stack(starting_channels, conv_layers)
        starting_neurons = self._compute_flattened_size(starting_channels)
        self.fc_stack = SimpleCNN._make_fc_stack(
            starting_neurons, lin_layers, num_classes
        )

    def _compute_flattened_size(self, starting_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, starting_channels, 32, 32
            )  # Example input size: (1, C, H, W)
            output = self.conv_stack(dummy_input)
            flattened_size = output.view(1, -1).size(1)
        return flattened_size

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x
