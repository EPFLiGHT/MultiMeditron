import torch
import torch.nn as nn
from typing import List, Optional

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        """
        Cross-attention module where queries attend over multiple context tensors (experts).

        ### Args
        * dim:         input feature dimension
        * num_heads:   number of attention heads
        * qkv_bias:    if True, add bias to Q, K, V projections
        * attn_drop:   dropout rate on attention weights
        * proj_drop:   dropout rate on output projection

        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        dim = int(dim)
        assert dim > 0, "dim must be positive"
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) # query proj 
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias) # key proj
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias) # value proj

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # [B, T, C] -> [B, num_heads, T, head_dim]
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,                        # [B, N_q, C] (queries)
        experts: List[torch.Tensor],           # list of [B, N_i, C] (contexts)
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, N_q, N_kv] or broadcastable
    ) -> torch.Tensor:
        """
        ### Args
        * x:        query states [B, N_q, C]
        * experts:  list of context tensors; each is [B, N_i, C]
                    we concat them along the sequence dimension -> [B, sum_i N_i, C]
        * attention_mask: standard additive or bool mask over KV positions.
        ### Returns
        * attended output [B, N_q, C]
        """
        B, N_q, C = x.shape

        # concat experts along sequence dim -> [B, N_kv, C]
        context = torch.cat(experts, dim=1)
        _, N_kv, _ = context.shape

        # project to Q, K, V: [B, T, C]
        q = self.q_proj(x)         # [B, N_q, C]
        k = self.k_proj(context)   # [B, N_kv, C]
        v = self.v_proj(context)   # [B, N_kv, C]

        # reshape to multi-head: [B, h, T, d]
        q = self._shape(q, B, N_q)     # [B, h, N_q, d]
        k = self._shape(k, B, N_kv)    # [B, h, N_kv, d]
        v = self._shape(v, B, N_kv)    # [B, h, N_kv, d]

        # scaled dot-product attention: [B, h, N_q, N_kv]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # boolean mask or additive mask
            if attention_mask.dtype == torch.bool:
                # if boolean, convert to additive:
                attn = attn.masked_fill(~attention_mask, float('-inf'))
            else:
                attn = attn + attention_mask

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # attention output: [B, h, N_q, d]
        out = torch.matmul(attn, v)

        # back to [B, N_q, C]
        out = out.transpose(1, 2).reshape(B, N_q, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
