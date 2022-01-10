"""
MobileNet v2:
https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
where H and W are expected to be at least 224.
The images have to be loaded in to a range of [0, 1] and then normalized
using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

MobileUnet
https://github.com/roeiherz/MobileUNET
"""

import math
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import InvertedResidual  # MobileNetV2


class MobileNetV2Unet(nn.Module):
    """MobileUnet or MobileNet v2 Unet"""

    def __init__(self, classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.classes = classes

        self.backbone = torch.hub.load(
            'pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(
            in_channels=3, out_channels=self.classes, kernel_size=1)
        
        self._init_weights()
    
    def _forward_features(self, frange: Tuple[int, int], x):
        for n in range(*frange):
            x = self.backbone.features[n](x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = x1 = self._forward_features((0, 2), x)
        logging.debug((x1.shape, 'x1'))

        x = x2 = self._forward_features((2, 4), x)
        logging.debug((x2.shape, 'x2'))

        x = x3 = self._forward_features((4, 7), x)
        logging.debug((x3.shape, 'x3'))

        x = x4 = self._forward_features((7, 14), x)
        logging.debug((x4.shape, 'x4'))

        x = x5 = self._forward_features((14, 19), x)
        logging.debug((x5.shape, 'x5'))

        # ----

        up1 = torch.cat([x4, self.dconv1(x)], dim=1)
        up1 = self.invres1(up1)
        logging.debug((up1.shape, 'up1'))

        up2 = torch.cat([x3, self.dconv2(up1)], dim=1)
        up2 = self.invres2(up2)
        logging.debug((up2.shape, 'up2'))

        up3 = torch.cat([x2, self.dconv3(up2)], dim=1)
        up3 = self.invres3(up3)
        logging.debug((up3.shape, 'up3'))

        up4 = torch.cat([x1, self.dconv4(up3)], dim=1)
        up4 = self.invres4(up4)
        logging.debug((up4.shape, 'up4'))

        x = self.conv_last(up4)
        logging.debug((x.shape, 'conv_last'))

        x = self.conv_score(x)
        logging.debug((x.shape, 'conv_score'))

        # ---

        # x = torch.sigmoid(x)
        # x = F.log_softmax(x)
        return x
