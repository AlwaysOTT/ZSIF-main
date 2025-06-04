from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat, einsum

from timm.models.layers import trunc_normal_ as __call_trunc_normal_, drop_path, to_2tuple
import torch.nn.functional as F

from src.models.ZSIF_modules.masking_generator import TubeMaskingGenerator


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class VideoTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, num_classes=0, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0., use_checkpoint=False, use_learnable_pos_emb=False, num_patches=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        # print(x.shape)
        x_vis = x
        if mask is not None:
            B, _, C = x.shape
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        # x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        # x = self.head(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x): #[B, 1536, 384]
        # print(f"x:{x.shape}")
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)) #[1152]
        # print(f"qkv_bias:{qkv_bias.shape}")
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) #[b,1536,1152]
        # print(f"qkv:{qkv.shape}")
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # print(f"qkv1:{qkv.shape}")
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) [b,head,1536,64]
        # print(f"q:{q.shape}")
        # print(f"k:{k.shape}")
        # print(f"v:{v.shape}")
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(f"attn:{attn.shape}")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # print(f"x:{x.shape}")
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(f"x1:{x.shape}")
        return x

class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=6,
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
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
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

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn

def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#
#     def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=512, num_frames=48, tubelet_size=16):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.tubelet_size = int(tubelet_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
#                 num_frames // tubelet_size) * in_chans
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.proj = nn.Conv3d(in_channels=1, out_channels=embed_dim,
#                               kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
#                               stride=(self.tubelet_size, patch_size[0], patch_size[1]))
#
#     def forward(self, x, **kwargs):
#         B, C, T, H, W = x.shape
#         # print(f"x1:{x.shape}")
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         # x = self.proj(x)  # b,384(dim),24,8,8
#         # x = x.flatten(2)  # b,384,1536
#         # x = x.transpose(1, 2)  # b,1536,384
#         m_list = []
#         for i in range(C):
#             m = x[:, i, :, :, :].unsqueeze(1)
#             m = self.proj(m)  # b,512,3,8,8
#             m = m.flatten(3)  # b,512,3,64
#             # print(f"m:{m.shape}")
#             m_list.append(m)
#         x = torch.cat(m_list, dim=-1)  # b,512,3,704
#         # print(f"x:{x.shape}")
#         x = x.flatten(2).transpose(1, 2)
#         # print(f"x:{x.shape}")
#         return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=512, num_frames=48, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0], patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


if __name__ == "__main__":
    x = torch.rand(2, 7, 48, 64, 64)
    patch = PatchEmbed(embed_dim=384)
    patch_x = patch(x)
    # model = VideoTransformer(num_patches=patch.num_patches)
    # m = TubeMaskingGenerator((16, 8, 8), 0.9)
    # m1 = torch.from_numpy(m())
    # m2 = torch.from_numpy(m())
    # m = torch.cat([m1.unsqueeze(0), m2.unsqueeze(0)], dim=0)
    # m = m.flatten(1).to(torch.bool)
    # out = model(patch_x, None)
    # print(out.shape)
    # print(patch_x.shape)
