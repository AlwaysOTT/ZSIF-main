import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from src.models.ZSIF_modules.attention_modules import PreNorm, FeedForward


class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.layers = nn.ModuleList([])
        # self.norm = nn.LayerNorm(dim)
        self.value_embedding = nn.Linear(num_frames, dim)
        self.dropout = nn.Dropout(dropout)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        # PreNorm(dim, ConvForward(dim, 12, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, x_mark):
        """
        Args:
            x: Input tensor of shape [B, T, D]
        """
        # x += self.pos_embedding
        x = torch.cat([x, x_mark], axis=-1)
        x = x.permute(0, 2, 1)
        x = self.dropout(self.value_embedding(x))
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Attention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out

class ConvForward(nn.Module):
    def __init__(self, dim, channel, kernel=3, scale=3, dropout=0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.Convlayer = nn.ModuleList([ConvLayer(channel, channel, kernel, channel, 2 ** i, 2 ** i, dropout)
                          for i in range(scale)])

    def forward(self, x):
        out = 0
        for layer in self.Convlayer:
            out += layer(x)
        x = x + out
        return x


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, group=1, padding=1, dilation=1, dropout=0):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.sampleConv = nn.Conv1d(in_channels=c_in,
                                    out_channels=c_out,
                                    groups=group,
                                    kernel_size=kernel,
                                    padding=padding,
                                    dilation=dilation)

    def forward(self, x):
        x = self.sampleConv(x)
        x = self.dropout(self.activation(x))
        return x


if __name__ == "__main__":
    device = torch.device('cuda:{}'.format(0))
    q = torch.rand(2, 12, 512).to(device)
    model = ConvForward(512, 12, 3).to(device)
    y = model(q)
    print(y.shape)
    # q = torch.rand(2, 2, 3, 2)
    # k = torch.rand(2, 2, 3, 2)
    # dots = einsum("b i h d, b j h d -> b h i j", q, k)
    # print(dots.shape)
    # print(dots)
    # einsum("b h i d, b h j d -> b h i j", q, k)
    # print(dots.shape)
    # print(dots)
    # x = torch.rand(2, 4, 3)
    # x_mark = torch.rand(2, 4, 3)
    # model = Transformer(512,4,4,6,64,2048,0)
    # y = model(x,x_mark)
    # print(y.shape)