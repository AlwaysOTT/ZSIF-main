from typing import Union, Tuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CrossPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, ctx, src_pos_emb, ts, tgt_pos_emb):
        return self.fn(self.norm_src(ctx), src_pos_emb, self.norm_tgt(ts), tgt_pos_emb)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.0,
            use_rotary=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, pos_emb):
        """
        Args:
            x: Sequence of shape [B, N, D]
            pos_emb: Positional embedding of sequence's tokens of shape [B, N, D]
        """

        q = self.to_q(x)

        qkv = (q, *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        if self.use_rotary:
            sin, cos = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), pos_emb
            )
            dim_rotary = sin.shape[-1]

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.0,
            use_rotary=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, src, src_pos_emb, tgt, tgt_pos_emb):

        q = self.to_q(tgt)

        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        if self.use_rotary:
            # apply 2d rotary embeddings to queries and keys
            if src_pos_emb is not None:
                sin_src, cos_src = map(
                    lambda t: repeat(t, "b n d -> b n h d", h=self.heads), src_pos_emb
                )
            if tgt_pos_emb is not None:
                sin_tgt, cos_tgt = map(
                    lambda t: repeat(t, "b n d -> b n h d", h=self.heads), tgt_pos_emb
                )
                dim_rotary = sin_tgt.shape[-1]
            # print(f"s:{sin_tgt.shape}")
            # print(f"s:{cos_tgt.shape}")

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            # print(f"q:{q.shape}")
            # print(f"q_pass:{q_pass.shape}")
            if tgt_pos_emb is not None:
                q = (q * cos_tgt) + (rotate_every_two(q) * sin_tgt)
            if src_pos_emb is not None:
                k = (k * cos_src) + (rotate_every_two(k) * sin_src)
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
        # print(f"q:{q.shape}")
        # print(f"k:{k.shape}")
        # print(f"v:{v.shape}")
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h i d -> b i (h d)", h=self.heads)
        return self.to_out(out), attn


class FilterAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.0,
            use_rotary=True,
            filter_size=6
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q_filter = nn.Linear(dim, inner_dim, bias=False)

        self.to_q_tgt = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv1 = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_kv2 = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out1 = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.to_out2 = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.filter = nn.Parameter(torch.randn(1, filter_size, dim))
        nn.init.normal_(self.filter, mean=0, std=0.02)

    def forward(self, src, src_pos_emb, tgt, tgt_pos_emb):
        q_filter = self.to_q_filter(self.filter)
        q_tgt = self.to_q_tgt(tgt)

        qkv = (q_filter, q_tgt, *self.to_kv1(src).chunk(2, dim=-1))
        # print(f"self.filter:{self.filter.shape}")
        q_filter, q_tgt, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        # print(f"q_src:{q_src.shape}")
        # print(f"k:{k.shape}")
        dots = einsum("b h i d, b h j d -> b h i j", q_filter, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads)
        out = self.to_out1(out)

        kv = self.to_kv2(out).chunk(2, dim=-1)
        k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), kv
        )
        dots = einsum("b h i d, b h j d -> b h i j", q_tgt, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads)

        return self.to_out2(out)



if __name__ == "__main__":
    x = torch.rand(16, 1024, 384)
    y = torch.rand(16, 48, 384)
    src = torch.rand(288, 64, 64)
    src_t = (src, src)
    tgt = torch.rand(288, 1, 64)
    tgt_t = (tgt, tgt)
    filter = torch.rand(1, 4, 384)
    filter = filter.expand(x.shape[0], -1, -1)
    model = FilterAttention(384, 6, 64, 0, False)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {} M'.format(n_parameters / 1e6))
    out, new_filter = model(x, src_t, y, tgt_t, filter)
    print(out.shape)
    print(new_filter.shape)

    # mp.spawn(main, nprocs=2)
