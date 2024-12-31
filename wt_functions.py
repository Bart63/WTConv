"""Input tensor X undergoes a cascade decomposition into different frequency bands
Used filter is made from 2D Haar wavelet
Kernels:
f_{LL} = 1/2 * [[1, 1], [1, 1]] - low-pass filter
f_{LH} = 1/2 * [[1, -1], [1, -1]]
f_{HL} = 1/2 * [[1, 1], [-1, -1]]
f_{HH} = 1/2 * [[1, -1], [-1, 1]]

The low-pass filter's response is then used for next decomposition (next level)
Each level increases frequency resolution while reducing spatial resolution
"""

import torch
import torch.nn.functional as F

# Haar wavelet filters
f_LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
f_LH = torch.tensor([[0.5, -0.5], [0.5, -0.5]]).unsqueeze(0).unsqueeze(0)
f_HL = torch.tensor([[0.5, 0.5], [-0.5, -0.5]]).unsqueeze(0).unsqueeze(0)
f_HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).unsqueeze(0).unsqueeze(0)


def WT(X):
    """
    Wavelet Transform (WT) for 2D images.

    Args:
        X (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        Four tensors of shape (B, C, H/2, W/2), each corresponding to a frequency band
    """

    _, C, _, _ = X.shape

    # Conv2d weights shape: (out_channels, in_channels/groups, kernel_height, kernel_width)
    depthwise_filters_LL = f_LL.repeat(C, 1, 1, 1)
    depthwise_filters_LH = f_LH.repeat(C, 1, 1, 1)
    depthwise_filters_HL = f_HL.repeat(C, 1, 1, 1)
    depthwise_filters_HH = f_HH.repeat(C, 1, 1, 1)

    X_LL = F.conv2d(X, depthwise_filters_LL, stride=2, groups=C)
    X_LH = F.conv2d(X, depthwise_filters_LH, stride=2, groups=C)
    X_HL = F.conv2d(X, depthwise_filters_HL, stride=2, groups=C)
    X_HH = F.conv2d(X, depthwise_filters_HH, stride=2, groups=C)
    return X_LL, X_LH, X_HL, X_HH


def IWT(X_LL, X_LH, X_HL, X_HH):
    """
    Inverse Wavelet Transform (IWT) for 2D images.

    Args:
        X_LL (torch.Tensor): Input tensor of shape (B, C, H/2, W/2).
        X_LH (torch.Tensor): Input tensor of shape (B, C, H/2, W/2).
        X_HL (torch.Tensor): Input tensor of shape (B, C, H/2, W/2).
        X_HH (torch.Tensor): Input tensor of shape (B, C, H/2, W/2).

    Returns:
        The reconstructed input tensor of shape (B, C, H, W).
    """

    _, C, _, _ = X_LL.shape

    depthwise_filters_LL = f_LL.repeat(C, 1, 1, 1)
    depthwise_filters_LH = f_HL.repeat(C, 1, 1, 1)
    depthwise_filters_HL = f_HL.repeat(C, 1, 1, 1)
    depthwise_filters_HH = f_HH.repeat(C, 1, 1, 1)

    X_reconstructed_LL = F.conv_transpose2d(X_LL, depthwise_filters_LL, stride=2, groups=C)
    X_reconstructed_LH = F.conv_transpose2d(X_LH, depthwise_filters_LH, stride=2, groups=C)
    X_reconstructed_HL = F.conv_transpose2d(X_HL, depthwise_filters_HL, stride=2, groups=C)
    X_reconstructed_HH = F.conv_transpose2d(X_HH, depthwise_filters_HH, stride=2, groups=C)

    X = X_reconstructed_LL + X_reconstructed_LH + X_reconstructed_HL + X_reconstructed_HH
    return X
