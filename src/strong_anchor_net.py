import torch
import torch.nn as nn


class DeconvHead(nn.Module):
    def __init__(self, in_ch=512, num_joints=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_joints, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class StrongAnchorNet(nn.Module):
    def __init__(self, num_joints=8, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                try:
                    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
                    print('Loaded pretrained ResNet18 weights.')
                except Exception as e:
                    print('Could not load pretrained weights, using random init:', repr(e))
                    backbone = resnet18(weights=None)
            else:
                backbone = resnet18(weights=None)
        except Exception as e:
            raise ImportError('torchvision is required for StrongAnchorNet. Install torchvision.') from e

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = DeconvHead(in_ch=512, num_joints=num_joints)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
