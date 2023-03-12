"""
Define model as a simplified version of CenterNet
"""

import torch
from torch import nn
import torchvision
import cv2


class centernet(nn.Module):
    """
    Define centernet variant, with convolutions up to original dimensions / 8
    Use main heatmap and 2 offset heatmaps as neck.
    """

    def __init__(self):
        super(centernet, self).__init__()

        # Resnet-18 as backbone.
        basemodel = torchvision.models.resnet18(weights=None)

        # Select only first layers up when you reach 160x90 dimensions with 128 channels
        self.base_model = nn.Sequential(*list(basemodel.children())[:-4])

        num_ch = 128
        head_conv = 64
        self.outc = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1),
        )

        self.outo = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # [b, 3, 720, 1280]

        x = self.base_model(x)
        # [b, 128, 90, 160]

        assert not torch.isnan(x).any()

        outc = self.outc(x)
        # [b, 1, 90, 160]
        assert not torch.isnan(outc).any()

        outo = self.outo(x)
        # [b, 2, 90, 160]
        assert not torch.isnan(outo).any()

        return outc, outo


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )
