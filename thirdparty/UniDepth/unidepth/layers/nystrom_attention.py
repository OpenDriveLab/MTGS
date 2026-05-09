from functools import partial
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from xformers.components.attention import NystromAttention as _XFormersNystromAttention
except ImportError:
    _XFormersNystromAttention = None

from .attention import AttentionBlock


class NystromBlock(AttentionBlock):
    _warned_missing_impl = False

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        self.attention_fn = None
        if _XFormersNystromAttention is not None:
            self.attention_fn = _XFormersNystromAttention(
                num_landmarks=128, num_heads=num_heads, dropout=dropout
            )
        elif not NystromBlock._warned_missing_impl:
            warnings.warn(
                'xformers.components.attention.NystromAttention is unavailable; '
                'falling back to standard scaled dot-product attention.',
                RuntimeWarning,
                stacklevel=2,
            )
            NystromBlock._warned_missing_impl = True

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        pos_embed_context: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if self.attention_fn is None:
            return super().attn(
                x,
                attn_bias=attn_bias,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
                rope=rope,
            )

        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x
