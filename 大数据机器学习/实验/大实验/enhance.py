import torch
import torch.nn as nn


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.adaptive_weight = nn.Parameter(torch.ones(num_features))
        self.adaptive_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        weight = self.adaptive_weight.view(1, -1, 1, 1)
        bias = self.adaptive_bias.view(1, -1, 1, 1)
        return weight * self.bn(x) + bias


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def replace_bn_with_adabn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, AdaptiveBatchNorm2d(module.num_features))
        elif isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                if isinstance(module, nn.BatchNorm2d):
                    setattr(
                        module,
                        child_name,
                        AdaptiveBatchNorm2d(child_module.num_features),
                    )


def add_seblock_to_resnet(model, reduction=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Conv2d):
                    # 在卷积层后添加 SEBlock
                    se_block = SEBlock(child_module.out_channels, reduction)
                    module.add_module(f"se_{child_name}", se_block)
