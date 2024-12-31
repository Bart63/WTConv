"""WTConv (Wavelet Transform Convolution) layer.

Input shape: (B, C, H, W).
Output shape: (B, C, H, W).
Based on: https://arxiv.org/pdf/2407.05848
"""

import torch
import torch.nn as nn

from wt_functions import WT, IWT


class WTConv(nn.Module):
    def __init__(self, in_channels, levels=0):
        """Initializes the WTConv layer.

        Args:
            in_channels (int): The number of input channels. Must be greater than 0.
            levels (int, optional): The number of decomposition levels.
                If 0, no decomposition is performed. If greater than 0, the input is decomposed
                into 4 frequency bands at each level. Default is 0.

        Raises:
            AssertionError: If `in_channels` is not greater than 0, or if `levels` is negative.

        Initializes a series of depth-wise convolutional layers, each corresponding to a
        decomposition level. Each level beyond 0 has 4 times the number of channels compared
        to the input, due to decomposition into 4 frequency bands.
        """

        assert in_channels > 0, 'Conv2d: number of input channels must be > 0'
        assert levels >= 0, 'WTConv: number of levels must be >= 0'

        super(WTConv, self).__init__()

        # Depth-wise convolution: groups=in_channels=out_channels
        # Each channel has its own filter
        # level=0 does not decompose
        # level>0 decompose into 4 responses with fixed number of channels
        self.levels = levels
        self.convs = nn.ModuleList()
        for i in range(self.levels + 1):
            nb_channels = in_channels if i == 0 else 4 * in_channels
            self.convs.append(
                nn.Conv2d(
                    in_channels=nb_channels,
                    out_channels=nb_channels,
                    kernel_size=3,
                    padding=1,
                    groups=nb_channels
                )
            )

    def forward(self, X):
        """
        Computes the output of the WTConv layer.

        The input is first decomposed into 4 frequency bands
        with Wavelet Transform (WT) at each level (except for
        the zero-th level, which doesn't decompose). Then,
        the input is processed by a depth-wise convolutional layer.
        Finally, the output is reconstructed by Inverse Wavelet Transform (IWT)
        and added back to the previous level.

        Args:
            X (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            The output tensor of shape (B, C, H, W).
        """

        X_LL = X                        # X^{0}_{LL}
        Y_0_LL = self.convs[0](X_LL)

        # No decomposition
        if self.levels == 0:
            return Y_0_LL

        Ys = []
        for conv in self.convs[1:]:
            X_LL, X_LH, X_HL, X_HH = WT(X_LL)
            Y_concat = conv(torch.cat((X_LL, X_LH, X_HL, X_HH), dim=1))
            Y_LL, Y_LH, Y_HL, Y_HH = torch.split(Y_concat, Y_concat.shape[1] // 4, dim=1)
            Ys.append([Y_LL, Y_LH, Y_HL, Y_HH])

        Z_l = torch.zeros_like(Ys[-1][0])           # Z^{l+1}
        for Y_LL, Y_LH, Y_HL, Y_HH in reversed(Ys):
            Z_l = IWT(Y_LL + Z_l, Y_LH, Y_HL, Y_HH)
        Z_l = Z_l + Y_0_LL                          # Z^{0}
        return Z_l


# Test WTConv with random input

if __name__ == '__main__':
    WTConv(3, 3)
    print('WTConv init: OK')

    X = torch.rand(1, 3, 32, 32)

    Y = WTConv(3, 0)(X)
    print('WTConv (layers=0) forward: OK')

    Y = WTConv(3, 3)(X)
    print('WTConv (layers=3) forward: OK')

    # Backward pass
    Y.backward(torch.ones_like(Y))
    print('WTConv backward: OK')