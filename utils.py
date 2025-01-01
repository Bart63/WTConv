import torch.nn.functional as F


def add_tensors_with_pad(X, Y):
    """
    Adds two tensors X and Y with different shapes by padding the smaller tensor.
    Shape: (B, C, H, W). Allowed differences only for H and W dimensions.

    Args:
        X (torch.Tensor): The larger tensor.
        Y (torch.Tensor): The smaller tensor.

    Returns:
        torch.Tensor: The sum of X and Y
    """
    _, _, H_x, W_x = X.shape
    _, _, H_y, W_y = Y.shape

    pad_h = H_x - H_y
    pad_w = W_x - W_y
    Y_padded = F.pad(Y, (0, pad_w, 0, pad_h))

    Z = X + Y_padded
    return Z
