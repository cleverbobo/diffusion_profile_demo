"""扩散模型 UNet 的核心 block (DDPM 风格, PyTorch)

对应结构图中的图元:
    - TimeEmbedding   : 正弦位置编码 + MLP，注入时间步 t
    - ResidualBlock   : GroupNorm-SiLU-Conv x2 + 时间步注入 + 残差 (图中 Residual)
    - AttentionBlock  : GroupNorm + 自注意力 + 残差       (图中 Attn)
    - DownSample      : stride=2 卷积下采样               (绿色箭头)
    - UpSample        : 最近邻上采样 + 卷积                 (紫色箭头)

DownBlock = ResidualBlock (+ AttentionBlock)
UpBlock   = ResidualBlock (+ AttentionBlock)，输入含 skip concat
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """正弦时间步编码 -> MLP，输出维度 dim。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        # 正弦编码恒为 fp32，纯半精度(非 autocast)时需对齐到 mlp 权重 dtype，
        # 否则 fp32 激活喂进 bf16/fp16 Linear 会报 dtype 不匹配；对 fp32 为 no-op。
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """GroupNorm-SiLU-Conv 两段 + 时间步注入 + 残差连接。"""

    def __init__(self, in_ch, out_ch, temb_ch, dropout=0.1, groups=32):
        super().__init__()
        g_in = math.gcd(groups, in_ch)
        g_out = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """空间自注意力 + 残差 (对应图中 Attn 部分)。"""

    def __init__(self, ch, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.ch = ch

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        # (b, hw, c)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w).permute(0, 2, 1)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class DownSample(nn.Module):
    """stride=2 卷积下采样。"""

    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):
    """最近邻插值上采样 + 卷积。"""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


if __name__ == "__main__":
    b, ch, temb_ch = 2, 128, 256
    x = torch.randn(b, ch, 16, 16)
    t = torch.randint(0, 1000, (b,))

    temb = TimeEmbedding(64)(t)  # -> (b, 256)
    print(f"TimeEmbedding: t{tuple(t.shape)} -> {tuple(temb.shape)}")

    res = ResidualBlock(ch, ch, temb_ch)
    print(f"ResidualBlock: {tuple(x.shape)} -> {tuple(res(x, temb).shape)}")

    attn = AttentionBlock(ch)
    print(f"AttentionBlock:{tuple(x.shape)} -> {tuple(attn(x).shape)}")

    print(f"DownSample:    {tuple(x.shape)} -> {tuple(DownSample(ch)(x).shape)}")
    print(f"UpSample:      {tuple(x.shape)} -> {tuple(UpSample(ch)(x).shape)}")
