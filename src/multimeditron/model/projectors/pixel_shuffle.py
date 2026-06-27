import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleProjector(nn.Module):
    """Projects patch embeddings to LLM hidden size via pixel-unshuffle spatial compression.

    Takes ``(batch, H*W, in_channels)`` patch embeddings laid out on a square grid
    and applies pixel-unshuffle at the given ``factor``, reducing the token count by
    ``factor**2`` while multiplying channels by ``factor**2``, then projects to
    ``out_channels`` via a single linear layer.

    Example for SigLIP2-512 (32×32 patches, 768 ch) with factor=4:
        Input:  (B, 1024, 768)
        Reshape: (B, 768, 32, 32)
        Unshuffle(4): (B, 12288, 8, 8)
        Reshape: (B, 64, 12288)
        Linear: (B, 64, out_channels)

    Args:
        in_channels: CLIP embedding dimension.
        out_channels: LLM hidden dimension.
        factor: Spatial downscale factor. Token count is reduced by ``factor**2``.
        dtype: Optional torch dtype for the linear layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(in_channels * factor * factor, out_channels, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings of shape ``(batch, H*W, in_channels)`` where H==W.

        Returns:
            Projected tensor of shape ``(batch, (H//factor)*(W//factor), out_channels)``.
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Expected square patch grid, got {N} tokens (not a perfect square)"
        assert H % self.factor == 0, (
            f"Spatial dim {H} must be divisible by pixel-shuffle factor {self.factor}"
        )
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)           # (B, C, H, W)
        x = F.pixel_unshuffle(x, self.factor)                     # (B, C*f², H/f, W/f)
        H_new = H // self.factor
        W_new = W // self.factor
        x = x.permute(0, 2, 3, 1).reshape(B, H_new * W_new, -1)  # (B, N/f², C*f²)
        return self.proj(x)                                        # (B, N/f², out_channels)
