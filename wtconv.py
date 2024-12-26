# %%
import matplotlib.pyplot as plt
import einops
import torch
import torch.nn.functional as F

# %%
f_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0)
f_lh = torch.tensor([[0.5, -0.5], [0.5, -0.5]]).unsqueeze(0).unsqueeze(0) 
f_hl = torch.tensor([[0.5, 0.5], [-0.5, -0.5]]).unsqueeze(0).unsqueeze(0)
f_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).unsqueeze(0).unsqueeze(0)

# %%
print(f_ll)
print(f_lh)
print(f_hl)
print(f_hh)

# %%

def depthwise_WTconv(X):
    _, C, _, _ = X.shape
    depthwise_filters_ll = f_ll.repeat(C, 1, 1, 1)
    depthwise_filters_lh = f_lh.repeat(C, 1, 1, 1)
    depthwise_filters_hl = f_hl.repeat(C, 1, 1, 1)
    depthwise_filters_hh = f_hh.repeat(C, 1, 1, 1)

    X_ll = F.conv2d(X, depthwise_filters_ll, stride=2, groups=C)
    X_lh = F.conv2d(X, depthwise_filters_lh, stride=2, groups=C)
    X_hl = F.conv2d(X, depthwise_filters_hl, stride=2, groups=C)
    X_hh = F.conv2d(X, depthwise_filters_hh, stride=2, groups=C)
    return X_ll, X_lh, X_hl, X_hh


def dephtwise_IWTConv(X_ll, X_lh, X_hl, X_hh):
    _, C, _, _ = X_ll.shape
    depthwise_filters_ll = f_ll.repeat(C, 1, 1, 1)
    depthwise_filters_lh = f_lh.repeat(C, 1, 1, 1)
    depthwise_filters_hl = f_hl.repeat(C, 1, 1, 1)
    depthwise_filters_hh = f_hh.repeat(C, 1, 1, 1)
    
    X_reconstructed_ll = F.conv_transpose2d(X_ll, depthwise_filters_ll, stride=2, groups=C)
    X_reconstructed_lh = F.conv_transpose2d(X_lh, depthwise_filters_lh, stride=2, groups=C)
    X_reconstructed_hl = F.conv_transpose2d(X_hl, depthwise_filters_hl, stride=2, groups=C)
    X_reconstructed_hh = F.conv_transpose2d(X_hh, depthwise_filters_hh, stride=2, groups=C)

    X = X_reconstructed_ll + X_reconstructed_lh + X_reconstructed_hl + X_reconstructed_hh
    return X

# %%

X = torch.ones((1, 3, 10, 10))
X[:, :, ::3] = 0
# %%
plt.imshow(einops.rearrange(X, 'b c h w -> h w (b c)'))

# %%
X_ll, X_lh, X_hl, X_hh = depthwise_WTconv(X)

# %%
X_recon = dephtwise_IWTConv(X_ll, X_lh, X_hl, X_hh)

# %%
plt.imshow(einops.rearrange(X_recon, 'b c h w -> h w (b c)'))

# %%
X_hh.max()