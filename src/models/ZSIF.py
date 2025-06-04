"""
Adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/rvt.py
"""
import random
from typing import List, Union, Tuple, OrderedDict
import torch
from torch import nn
from src.models.ZSIF_modules.CrossTransformer import CrossTransformer
from src.models.ZSIF_modules.Transformer import Transformer
from src.models.ZSIF_modules.VideoTransformer import VideoTransformer, PatchEmbed
from src.utils.tools import my_load_state_dict



class Model(nn.Module):
    def __init__(
            self,
            image_size: Union[List[int], Tuple[int]],
            patch_size: Union[List[int], Tuple[int]],
            # time_coords_encoder: nn.Module,
            dim: int = 512,
            depth: int = 4,
            heads: int = 6,
            mlp_ratio: int = 4,
            ctx_channels: int = 5,
            ts_channels: int = 3,
            ts_length: int = 48,
            out_dim: int = 1,
            dim_head: int = 64,
            dropout: float = 0.2,
            freq_type: str = "lucidrains",
            pe_type: str = "learned",
            num_mlp_heads: int = 1,
            use_glu: bool = True,
            ctx_masking_ratio: float = 0,
            ts_masking_ratio: float = 0,
            decoder_dim: int = 128,
            decoder_depth: int = 4,
            decoder_heads: int = 6,
            decoder_dim_head: int = 128,
            ctx_dim: int = 512,
            ctx_depth: int = 8,
            ctx_tubelet: int = 6,
            ctx_dropout=0.2,
            ctx_droppath=0,
            mix_dim: int = 512,
            mix_depth: int = 4,
            mix_heads: int = 6,
            mix_mlp_ratio: int = 4,
            mix_dim_head: int = 64,
            mix_dropout: int = 0.2,
            use_pretrain: bool = False,
            pretrain_ck: str = "",
            **kwargs,
    ):
        super().__init__()
        assert (
                ctx_masking_ratio >= 0 and ctx_masking_ratio < 1
        ), "ctx_masking_ratio must be in [0,1)"
        assert pe_type in [
            "rope",
            "sine",
            "learned",
            None,
        ], f"pe_type must be 'rope', 'sine', 'learned' or None but you provided {pe_type}"
        # self.time_coords_encoder = time_coords_encoder
        # self.ctx_channels = ctx_channels
        # self.ts_channels = ts_channels
        # if hasattr(self.time_coords_encoder, "dim"):
        #     self.ctx_channels += self.time_coords_encoder.dim
        #     self.ts_channels += self.time_coords_encoder.dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.ctx_masking_ratio = ctx_masking_ratio
        self.ts_masking_ratio = ts_masking_ratio
        self.num_mlp_heads = num_mlp_heads
        self.pe_type = pe_type

        # self.dct_layer = dct_channel_block(256)

        for i in range(2):
            ims = self.image_size[i]
            ps = self.patch_size[i]
            assert (
                    ims % ps == 0
            ), "Image dimensions must be divisible by the patch size."

        # patch_dim = self.ctx_channels * self.patch_size[0] * self.patch_size[1]
        # num_patches = (self.image_size[0] // self.patch_size[0]) * (
        #         self.image_size[1] // self.patch_size[1]
        # )

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange(
        #         "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        #         p1=self.patch_size[0],
        #         p2=self.patch_size[1],
        #     ),
        #     nn.Linear(patch_dim, ctx_dim),
        # )
        # self.enc_pos_emb = AxialRotaryEmbedding(dim_head, freq_type, **kwargs)
        # self.ts_embedding = nn.Linear(self.ts_channels, dim)
        self.to_patch_embedding = PatchEmbed(embed_dim=ctx_dim, in_chans=ctx_channels, tubelet_size=ctx_tubelet)
        self.ctx_encoder = VideoTransformer(
            embed_dim=ctx_dim,
            depth=ctx_depth,
            drop_rate=ctx_dropout,
            attn_drop_rate=ctx_dropout,
            drop_path_rate=ctx_droppath,
            num_patches=self.to_patch_embedding.num_patches)
        # self.ctx_encoder = VisionTransformer(
        #     ctx_dim,
        #     ctx_depth,
        #     ctx_heads,
        #     ctx_dim_head,
        #     ctx_dim * ctx_mlp_ratio,
        #     image_size,
        #     ctx_dropout,
        #     pe_type == "rope",
        #     use_glu,
        # )

        if use_pretrain:
            print("start load ctx_encoder...")
            checkpoint = torch.load(pretrain_ck, map_location='cpu')
            all_keys = list(checkpoint["model"].keys())
            # patch_dict = OrderedDict()
            # for key in all_keys:
            #     if key.startswith('patch_embed.'):
            #         patch_dict[key[12:]] = checkpoint["model"][key]
            # my_load_state_dict(self.to_patch_embedding, patch_dict, "")
            # for param in self.to_patch_embedding.parameters():
            #     param.requires_grad = False
            model_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('encoder.'):
                    model_dict[key[8:]] = checkpoint["model"][key]
            my_load_state_dict(self.ctx_encoder, model_dict, "")
            # for param in self.ctx_encoder.parameters():
            #     param.requires_grad = False

        # 预训练!!!!
        # checkpoint = torch.load("/home/ma-user/work/code/Forecast-main_ST/ts_context_logs/ctx_model.pth")
        # model_dict = self.ctx_encoder.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.ctx_encoder.load_state_dict(model_dict)
        # for param in self.ctx_encoder.parameters():
        #     param.requires_grad = False

        # if pe_type == "learned":
        #     self.pe_ctx = nn.Parameter(torch.randn(1, self.to_patch_embedding.num_patches, ctx_dim))
            # self.pe_ts = nn.Parameter(torch.randn(1, 13, dim))
        # elif pe_type == "sine":
            # self.pe_ctx = PositionalEncoding2D(dim)
            # self.pe_ts = PositionalEncoding2D(dim)
        self.ts_encoder = Transformer(
            dim,
            ts_length,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            dropout=dropout,
        )

        self.mixer = CrossTransformer(
            mix_dim,
            mix_depth,
            mix_heads,
            mix_dim_head,
            mix_dim * mix_mlp_ratio,
            image_size,
            mix_dropout,
            pe_type == "rope",
            use_glu,
        )
        # self.ctx_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))


        # self.ts_enctodec = nn.Linear(dim, decoder_dim)
        # self.temporal_transformer = Transformer(
        #     decoder_dim,
        #     ts_length,
        #     decoder_depth,
        #     decoder_heads,
        #     decoder_dim_head,
        #     decoder_dim * mlp_ratio,
        #     dropout=dropout,
        # )
        self.projector = nn.Linear(dim, ts_length)
        # self.projector = nn.Linear(13 * dim, ts_length)
        # self.ts_mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # self.mlp_heads = nn.ModuleList([])
        # for i in range(num_mlp_heads):
        #     self.mlp_heads.append(
        #         nn.Sequential(
        #             nn.LayerNorm(decoder_dim),
        #             nn.Linear(decoder_dim, out_dim, bias=True),
        #             nn.ReLU(),
        #         )
        #     )

        # self.quantile_masker = nn.Sequential(
        #     nn.Conv1d(decoder_dim, dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     Rearrange(
        #         "b c t -> b t c",
        #     ),
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_mlp_heads),
        # )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # print(f"len:{len_keep}")

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(
            self,
            ctx: torch.Tensor,
            ctx_coords: torch.Tensor,
            ts: torch.Tensor,
            ts_coords: torch.Tensor,
            time_coords: torch.Tensor,
            mask: bool = True,
    ):
        """
        Args:
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference
        Returns:

        """
        B, T, _, H, W = ctx.shape
        _, _, N = ts.shape
        # time_coords = self.time_coords_encoder(time_coords)
        # 时序和云图加时间编码
        # ctx = torch.cat([ctx, time_coords], axis=2)
        # ts = torch.cat([ts, time_coords[..., 0, 0]], axis=-1)

        # ctx = rearrange(ctx, "b t c h w -> (b t) c h w")
        # print(f"ctx_coords:{ctx_coords.shape}")
        # ctx_coords = repeat(ctx_coords, "b c h w -> (b t) c h w", t=T)
        # print(f"ctx_coords1:{ctx_coords.shape}")
        # ts_coords = repeat(ts_coords, "b c h w -> (b t) c h w", t=T)
        # 坐标相对位置编码
        src_enc_pos_emb = None  # self.enc_pos_emb(ctx_coords)  # tuple((bt,64,64),(bt,64,64))
        tgt_pos_emb = None  # self.enc_pos_emb(ts_coords)  # tuple((bt,1,64),(bt,1,64))

        # 加坐标
        # ctx_coords = ctx_coords.unsqueeze(1).repeat(1, T, 1, 1, 1)
        # ts_coords = ts_coords[:, :, 0, 0].unsqueeze(1).repeat(1, T, 1)
        # ctx = torch.cat([ctx, ctx_coords], axis=2)
        # ts = torch.cat([ts, ts_coords], axis=2)
        # print(f"src_enc_pos_emb:{src_enc_pos_emb[0].shape}")
        # print(f"tgt_pos_emb:{tgt_pos_emb[0].shape}")
        ctx = ctx.permute(0, 2, 1, 3, 4)
        # print(f"ctx:{ctx.shape}")
        ctx = self.to_patch_embedding(ctx)  # B, 1536, 384
        # print(f"ctx1:{ctx.shape}")
        # if self.pe_type == "learned":
        #     ctx = ctx + self.pe_ctx
        # elif self.pe_type == "sine":
        #     pe = self.pe_ctx(ctx_coords)
        #     pe = rearrange(pe, "b h w c -> b (h w) c")
        #     ctx = ctx + pe
        if self.ctx_masking_ratio > 0 and mask:
            p = self.ctx_masking_ratio * random.random()
            # print(f"p:{p}")
            ctx, _, ids_restore, ids_keep = self.random_masking(ctx, p)  # 掩盖patch后ctx(bt,masked,384)
            src_enc_pos_emb = tuple(
                torch.gather(
                    pos_emb,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_emb.shape[-1]),
                )
                for pos_emb in src_enc_pos_emb
            )
        latent_ctx = self.ctx_encoder(ctx)  # b,1536,384
        # print(f"ctx2:{latent_ctx.shape}")

        # ts = self.ts_embedding(ts)  # b,t,384
        if self.ts_masking_ratio > 0 and mask:
            p = self.ts_masking_ratio * random.random()
            ts, _, ids_restore, ids_keep = self.random_masking(ts, p)
            mask_tokens = self.ts_mask_token.repeat(ts.shape[0], T - ts.shape[1], 1)
            ts = torch.cat([ts, mask_tokens], dim=1)
            ts = torch.gather(
                ts, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, ts.shape[2])
            )
        # print(ts.shape)

        latent_ts = self.ts_encoder(ts, time_coords[..., 0, 0])  # b,t,c
        # print(latent_ts.shape)
        # 增加DCT
        # latent_ts = self.dct_layer(latent_ts)

        # latent_ts = rearrange(latent_ts, "b t c -> (b t) c").unsqueeze(1)
        # print(f"latent_ts:{latent_ts.shape}")
        # 融合层加上坐标相对位置编码
        # if self.pe_type == "learned":
        #     latent_ctx = latent_ctx + self.pe_ctx
            # latent_ts = latent_ts + self.pe_ts
            # print(f"latent_ts1:{latent_ts.shape}")
        # elif self.pe_type == "sine":
        #     pe = self.pe_ts(ts_coords)
        #     pe = rearrange(pe, "b h w c -> b (h w) c")
        #     latent_ts = latent_ts + pe
        latent_ts, cross_attention_scores = self.mixer(
            latent_ctx, latent_ts, src_enc_pos_emb, tgt_pos_emb
        )  # bt,1,384
        # print(f"latent_ts:{latent_ts.shape}")
        # latent_ts = latent_ts.squeeze(1)
        # latent_ts = self.ts_enctodec(latent_ts)
        # print(f"latent_ts1:{latent_ts.shape}")

        y = self.projector(latent_ts).permute(0, 2, 1)[:, :, 0]
        # dim = latent_ts.shape[2]
        # y = self.projector(latent_ts.view(-1, 13 * dim))
        # y = self.temporal_transformer(latent_ts)
        # print(f"y:{y.shape}")
        # print(y)

        # Handles the multiple MLP heads
        # outputs = []
        # for i in range(self.num_mlp_heads):
        #     mlp = self.mlp_heads[i]
        #     output = mlp(y)
        #     outputs.append(output)
        # outputs = torch.stack(outputs, dim=2)
        # print(f"outputs:{outputs.shape}")

        # y_d = rearrange(y.detach(), "b t c -> b c t")
        # quantile_mask = self.quantile_masker(y_d)

        quantile_mask = {}

        return (y, quantile_mask, cross_attention_scores)


if __name__ == "__main__":
    # model = VideoTransformer(
    #     drop_rate=0.1,
    #     attn_drop_rate=0.1,
    #     drop_path_rate=0.4,
    #     num_patches=1536)
    # model_state = model.state_dict()
    # print(len(model_state))
    # for k in model_state.keys():
    #     print(k)
    # checkpoint = torch.load("./best.pth", map_location='cpu')
    # print(checkpoint.keys())
    # for k in checkpoint["model"].keys():
    #     print(k)
    # all_keys = list(checkpoint["model"].keys())
    # new_dict = OrderedDict()
    # for key in all_keys:
    #     if key.startswith('encoder.'):
    #         new_dict[key[8:]] = checkpoint["model"][key]
    # for k in new_dict.keys():
    #     print(k)
    # from src.utils.tools import my_load_state_dict

    # my_load_state_dict(model, new_dict, "")
    # print(checkpoint["model"])
    # print(checkpoint["args"])
    device = torch.device('cuda:{}'.format(0))
    model = Model(image_size=[64, 64], patch_size=[8, 8]).to(device)
    # checkpoint = torch.load("/home/zhangfeng/tangjh/oth/Forecast_pretrain/ts_context_logs/PTMM/2024-02-16_20-35-16/checkpoints/best.pth", map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {} M'.format(n_parameters / 1e6))

    # time_e = Cyclical_embedding([12, 31, 24, 60])
    # model = RoCrossViViT_bis(
    #     image_size=[64, 64],
    #     patch_size=[8, 8],
    #     # time_coords_encoder=time_e,
    #     dim=512,
    #     depth=4,
    #     heads=6,
    #     mlp_ratio=4,
    #     ctx_channels=5,
    #     ts_channels=8,
    #     ts_length=48,
    #     out_dim=1,
    #     dim_head=64,
    #     dropout=0.0,
    #     freq_type="lucidrains",
    #     pe_type="learned",
    #     num_mlp_heads=1,
    #     use_glu=True,
    #     ctx_masking_ratio=0,
    #     ts_masking_ratio=0,
    #     decoder_dim=128,
    #     decoder_depth=4,
    #     decoder_heads=6,
    #     decoder_dim_head=128,
    #     max_freq=128,
    #     ctx_dim=384,
    # ).to(device)
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params: {} M'.format(n_parameters / 1e6))
    ctx = torch.rand(2, 48, 5, 64, 64).to(device)
    ctx_coord = torch.rand(2, 2, 8, 8).to(device)
    ts = torch.rand(2, 48, 8).to(device)
    ts_coord = torch.rand(2, 2, 1, 1).to(device)
    time_coord = torch.rand(2, 48, 4, 64, 64).to(device)
    out = model(ctx, ctx_coord, ts, ts_coord, time_coord)
    print(out[0].shape)
    # print(out[0])
    # mlp_heads = nn.ModuleList([])
    # for i in range(1):
    #     mlp_heads.append(
    #         nn.Sequential(
    #             nn.LayerNorm(128),
    #             nn.Linear(128, 1, bias=True),
    #             nn.ReLU(),
    #         )
    #     )
    # x = torch.rand(2, 48, 128)
    # outputs = []
    # for i in range(1):
    #     mlp = mlp_heads[i]
    #     output = mlp(x)
    #     outputs.append(output)
    # print(output[0].shape)
