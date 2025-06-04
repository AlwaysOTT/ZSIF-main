import torch
from torch import nn
from typing import List, Tuple, Union

from src.models.ZSIF_modules.attention_modules import PreNorm, SelfAttention, FeedForward


class VisionTransformer(nn.Module):
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

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
            self,
            src: torch.Tensor,
            src_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
        Args:
            src: Source sequence of shape [B, N, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
        """

        attention_scores = {}
        for i in range(len(self.blocks)):
            sattn, sff = self.blocks[i]

            out, sattn_scores = sattn(src, pos_emb=src_pos_emb)
            attention_scores["self_attention"] = sattn_scores
            src = out + src
            src = sff(src) + src

        return src, attention_scores
