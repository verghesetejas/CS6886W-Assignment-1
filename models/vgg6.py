import torch.nn as nn

def activation(name):
    layers = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
              "silu": nn.SiLU, "gelu": nn.GELU}
    if name.lower() not in layers:
        raise ValueError(f"Unsupported activation: {name}")
    return layers[name.lower()]()

class VGG6(nn.Module):
    # Professor's configuration: [64, 64, 'M', 128, 128, 'M'].
    def __init__(self, num_classes=10, act="relu", batch_norm=True):
        super().__init__()
        blocks, in_ch = [], 3
        for item in [64, 64, "M", 128, 128, "M"]:
            if item == "M":
                blocks.append(nn.MaxPool2d(2, 2))
            else:
                blocks.append(nn.Conv2d(in_ch, item, 3, padding=1, bias=not batch_norm))
                if batch_norm:
                    blocks.append(nn.BatchNorm2d(item))
                blocks.append(activation(act))
                in_ch = item
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(self.pool(x).flatten(1))
