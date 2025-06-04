import torch
from torch import nn
from math import pi
from einops import rearrange, repeat


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, freq_type="lucidrains", **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_type = freq_type
        if freq_type == "lucidrains":
            scales = torch.linspace(1.0, kwargs["max_freq"] / 2, self.dim // 4)
        elif freq_type == "vaswani":
            scales = 1 / (
                    kwargs["base"] ** (torch.arange(0, self.dim, 4).float() / self.dim)
            )
        else:
            NotImplementedError(
                f"Only 'lucidrains' and 'vaswani' frequencies are implemented, but you chose {freq_type}."
            )
        self.register_buffer("scales", scales)

    def forward(self, coords: torch.Tensor):
        """
        Assumes that coordinates do not change throughout the batches.
        Args:
            coords (torch.Tensor): Coordinates of shape [B, 2, H, W]
        """
        seq_x = coords[:, 0, 0, :]
        # print(f"seq_x:{seq_x.shape}")
        seq_x = seq_x.unsqueeze(-1)
        # print(f"seq_x1:{seq_x.shape}")
        seq_y = coords[:, 1, :, 0]
        # print(f"seq_y:{seq_y.shape}")
        seq_y = seq_y.unsqueeze(-1)
        # print(f"seq_y1:{seq_y.shape}")

        scales = self.scales[(*((None, None)), Ellipsis)]
        scales = scales.to(coords)
        # print(f"scales:{scales.shape}")
        if self.freq_type == "lucidrains":
            seq_x = seq_x * scales * pi
            seq_y = seq_y * scales * pi
        elif self.freq_type == "vaswani":
            seq_x = seq_x * scales
            seq_y = seq_y * scales
        # print(f"seq_y2:{seq_y.shape}")
        x_sinu = repeat(seq_x, "b i d -> b i j d", j=seq_y.shape[1])
        y_sinu = repeat(seq_y, "b j d -> b i j d", i=seq_x.shape[1])
        # print(f"x_sinu:{x_sinu.shape}")
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)
        # print(f"sin:{sin.shape}")
        sin, cos = map(lambda t: rearrange(t, "b i j d -> b (i j) d"), (sin, cos))
        # print(f"sin1:{sin.shape}")
        sin, cos = map(lambda t: repeat(t, "b n d -> b n (d j)", j=2), (sin, cos))
        # print(f"sin2:{sin.shape}")
        return sin, cos

if __name__ == "__main__":
    model = AxialRotaryEmbedding(64,freq_type="lucidrains",max_freq=128)
    x = torch.rand(16, 2, 8, 8)
    out = model(x)
    print(out[0].shape)