from typing import Union, List, Tuple

import torch
from torch import nn

from src.models.ZSIF_modules.attention_modules import CrossPreNorm, CrossAttention, PreNorm, FeedForward, \
    FilterAttention


class CrossTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            image_size: Union[List[int], Tuple[int], int],
            dropout: float = 0.0,
            use_rotary: bool = True,
            use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        self.cross_layers = nn.ModuleList([])

        for _ in range(depth):
            self.cross_layers.append(
                nn.ModuleList(
                    [
                        CrossPreNorm(
                            dim,
                            FilterAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        )
                    ]
                )
            )
        # self.filter = nn.Parameter(torch.randn(1, 6, dim))
        # nn.init.normal_(self.filter, mean=0, std=0.02)
        # self.norm_filter = nn.LayerNorm(dim)

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_pos_emb: torch.Tensor,
            tgt_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
            3. Cross-Attention between target and source sequence
            4. FFN on the target sequence
        Args:
            src: Source sequence of shape [B, N, D]
            tgt: Target sequence of shape [B, M, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
            tgt_pos_emb: Positional embedding of target sequence's tokens of shape [B, M, D]
        """

        attention_scores = {}
        # expanded_filter = self.filter.expand(src.shape[0], -1, -1)
        for i in range(len(self.cross_layers)):

            cattn, cff = self.cross_layers[i]
            out = cattn(src, src_pos_emb, tgt, tgt_pos_emb)
            # print(f"filter:{filter.shape}")
            # attention_scores["cross_attention"] = cattn_scores
            tgt = out + tgt
            tgt = cff(tgt) + tgt
            # expanded_filter = self.norm_filter(filter + expanded_filter)

        return tgt, attention_scores
