"""GPU elastic deformation for data augmentation.

Replicates PyTorch's ElasticTransform on GPU with a custom CUDA separable
Gaussian blur kernel. No backward needed (augmentation only).

The Gaussian kernel formula matches PyTorch exactly:
  kernel1d = softmax(-(x/sigma)^2)  where x = linspace(-lim, lim, ksize)
  lim = (ksize - 1) / (2 * sqrt(2)),  ksize = int(8*sigma + 1) rounded to odd
"""

import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------
# Gaussian kernel computation (matches PyTorch's ElasticTransform exactly)
# ---------------------------------------------------------------------------

def _compute_elastic_kernel_1d(sigma: float) -> torch.Tensor:
    """Compute 1D Gaussian kernel matching PyTorch ElasticTransform.

    Returns: float32 CPU tensor [kernel_size].
    """
    kernel_size = int(8 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0))
    x = torch.linspace(-lim, lim, kernel_size)
    kernel_1d = torch.softmax(-(x / sigma) ** 2, dim=0)
    return kernel_1d


# ---------------------------------------------------------------------------
# Cached state (avoids recomputation across calls)
# ---------------------------------------------------------------------------

_cached_kernel_1d = None
_cached_kernel_2d_gpu = None  # for fallback path
_cached_sigma = None
_use_cuda_blur = None  # None = not yet probed


def _probe_cuda_blur():
    """Check if custom CUDA blur ops are available."""
    global _use_cuda_blur
    try:
        import asann_cuda_ops
        asann_cuda_ops.elastic_set_kernel_weights
        asann_cuda_ops.elastic_blur_separable
        _use_cuda_blur = True
    except (ImportError, AttributeError):
        _use_cuda_blur = False
    return _use_cuda_blur


def _get_kernel(sigma: float, device: torch.device):
    """Get/cache kernel weights. Uploads to CUDA constant memory if available."""
    global _cached_kernel_1d, _cached_kernel_2d_gpu, _cached_sigma, _use_cuda_blur

    if _use_cuda_blur is None:
        _probe_cuda_blur()

    if _cached_sigma == sigma and _cached_kernel_1d is not None:
        return _cached_kernel_1d

    kernel_1d = _compute_elastic_kernel_1d(sigma)
    _cached_kernel_1d = kernel_1d
    _cached_sigma = sigma

    if _use_cuda_blur:
        import asann_cuda_ops
        asann_cuda_ops.elastic_set_kernel_weights(kernel_1d.contiguous())

    # Pre-compute 2D kernel for fallback path
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # [K, K]
    _cached_kernel_2d_gpu = kernel_2d.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,K,K]

    return kernel_1d


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_elastic_deform(
    x_batch: torch.Tensor,
    spatial_shape: tuple,
    alpha: float = 30.0,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Apply elastic deformation to a batch of flat image tensors on GPU.

    Args:
        x_batch: [B, D_flat] tensor on GPU, already normalized.
        spatial_shape: (C, H, W) tuple.
        alpha: Displacement magnitude (pixels before normalization).
        sigma: Gaussian smoothing sigma (controls smoothness).

    Returns:
        [B, D_flat] tensor with elastic deformation applied.
    """
    C, H, W = spatial_shape
    B = x_batch.shape[0]
    imgs = x_batch.view(B, C, H, W)

    # Ensure kernel is ready
    kernel_1d = _get_kernel(sigma, x_batch.device)
    kernel_size = len(kernel_1d)

    # Step 1: Random displacement fields [B, 1, H, W] in [-1, 1]
    dx = torch.rand(B, 1, H, W, device=x_batch.device) * 2 - 1
    dy = torch.rand(B, 1, H, W, device=x_batch.device) * 2 - 1

    # Step 2: Gaussian blur the displacement fields
    if _use_cuda_blur:
        import asann_cuda_ops

        # Stack dx, dy as [2B, H, W] for batched blur
        disp = torch.cat([dx.squeeze(1), dy.squeeze(1)], dim=0).contiguous()
        blurred = asann_cuda_ops.elastic_blur_separable(disp, kernel_size)
        dx_blur = blurred[:B].unsqueeze(1)   # [B, 1, H, W]
        dy_blur = blurred[B:].unsqueeze(1)   # [B, 1, H, W]
    else:
        # Fallback: PyTorch conv2d with reflect padding (still on GPU)
        pad = kernel_size // 2
        dx_padded = F.pad(dx, [pad] * 4, mode='reflect')
        dy_padded = F.pad(dy, [pad] * 4, mode='reflect')
        dx_blur = F.conv2d(dx_padded, _cached_kernel_2d_gpu)
        dy_blur = F.conv2d(dy_padded, _cached_kernel_2d_gpu)

    # Step 3: Scale displacement by alpha / image_size
    dx_scaled = dx_blur.squeeze(1) * (alpha / H)  # [B, H, W]
    dy_scaled = dy_blur.squeeze(1) * (alpha / W)  # [B, H, W]

    # Step 4: Build sampling grid (identity + displacement)
    # Identity grid matching PyTorch's align_corners=False convention
    grid_y = torch.linspace((-H + 1) / H, (H - 1) / H, H,
                            device=x_batch.device, dtype=x_batch.dtype)
    grid_x = torch.linspace((-W + 1) / W, (W - 1) / W, W,
                            device=x_batch.device, dtype=x_batch.dtype)
    grid_yy, grid_xx = torch.meshgrid(grid_y, grid_x, indexing='ij')

    # Expand to [B, H, W] and add displacement
    grid_xx = grid_xx.unsqueeze(0).expand(B, -1, -1) + dx_scaled
    grid_yy = grid_yy.unsqueeze(0).expand(B, -1, -1) + dy_scaled

    # Stack to [B, H, W, 2] — grid_sample expects (x, y) order
    grid = torch.stack([grid_xx, grid_yy], dim=-1)

    # Step 5: Bilinear grid sampling (uses PyTorch's CUDA kernel)
    output = F.grid_sample(
        imgs, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    return output.reshape(B, -1)
