import torch
import torch.nn as nn

class PixelShuffleProjector(nn.Module):
    """
    A Pixel Shuffle (Space-to-Depth) projector that downsamples spatial tokens 
    while increasing channel dimension, followed by a linear projection.
    
    This matches the architecture used in nanoVLM to compress 1024 tokens into 64.
    """
    def __init__(self, modality_size: int, projected_size: int, factor: int = 4, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.factor = factor
        self.modality_size = modality_size
        self.projected_size = projected_size
        
        # After space-to-depth, the channel dimension becomes modality_size * (factor^2)
        self.projection = nn.Linear(modality_size * (factor * factor), projected_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, C) where L = H * W
        Returns:
            Projected tokens of shape (B, L/(factor^2), projected_size)
        """
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        if H * W != L:
            raise ValueError(f"Input sequence length {L} is not a perfect square.")
            
        # Reshape to (B, H, W, C)
        x = x.view(B, H, W, C)
        
        # Space-to-depth (Pixel Unshuffle)
        # We manually reshape and transpose to achieve space-to-depth
        # Shape: (B, H/f, f, W/f, f, C)
        f = self.factor
        x = x.view(B, H // f, f, W // f, f, C)
        # Permute to (B, H/f, W/f, f, f, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Flatten to (B, L_out, C * f^2)
        x = x.view(B, (H // f) * (W // f), C * f * f)
        
        # Project to LLM dimension
        return self.projection(x)

