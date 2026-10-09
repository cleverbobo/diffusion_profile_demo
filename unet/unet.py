"""DDPM UNet：用 blocks.py 中的 block 按结构图堆叠。

结构:
    Conv(in->base)
    ┌ 每个分辨率 level: num_res_blocks 个 (ResidualBlock[+Attn]) -> 存 skip
    │                   非最后一层再接 DownSample
    Middle: ResidualBlock -> AttentionBlock -> ResidualBlock
    └ 每个 level: (num_res_blocks+1) 个 (concat skip -> ResidualBlock[+Attn])
                  非最后一层再接 UpSample
    GroupNorm-SiLU-Conv(base->out)

默认配置对应结构图 (32x32x3, base=64, ch_mult=(1,2,4,8), 8x8 处加 attention)。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import (
    AttentionBlock,
    DownSample,
    ResidualBlock,
    TimeEmbedding,
    UpSample,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base_ch=64,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(8,),
        img_size=32,
        dropout=0.1,
    ):
        super().__init__()
        temb_ch = base_ch * 4
        self.time_embed = TimeEmbedding(base_ch)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ---------- Encoder ----------
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        cur_res = img_size
        num_levels = len(ch_mult)
        for i, mult in enumerate(ch_mult):
            out = base_ch * mult
            level = nn.ModuleList()
            for _ in range(num_res_blocks):
                blk = nn.ModuleList([ResidualBlock(ch, out, temb_ch, dropout)])
                if cur_res in attn_resolutions:
                    blk.append(AttentionBlock(out))
                level.append(blk)
                ch = out
                skip_chs.append(ch)
            self.down_blocks.append(level)
            if i != num_levels - 1:
                self.down_samples.append(DownSample(ch))
                skip_chs.append(ch)
                cur_res //= 2
            else:
                self.down_samples.append(None)

        # ---------- Middle ----------
        self.mid_res1 = ResidualBlock(ch, ch, temb_ch, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_res2 = ResidualBlock(ch, ch, temb_ch, dropout)

        # ---------- Decoder ----------
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out = base_ch * mult
            level = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blk = nn.ModuleList(
                    [ResidualBlock(ch + skip_chs.pop(), out, temb_ch, dropout)]
                )
                if cur_res in attn_resolutions:
                    blk.append(AttentionBlock(out))
                level.append(blk)
                ch = out
            self.up_blocks.append(level)
            if i != 0:
                self.up_samples.append(UpSample(ch))
                cur_res *= 2
            else:
                self.up_samples.append(None)

        g = math.gcd(32, ch)
        self.out_norm = nn.GroupNorm(g, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_embed(t)
        h = self.in_conv(x)
        skips = [h]

        # Encoder
        for level, down in zip(self.down_blocks, self.down_samples):
            for blk in level:
                h = blk[0](h, temb)
                if len(blk) > 1:
                    h = blk[1](h)
                skips.append(h)
            if down is not None:
                h = down(h)
                skips.append(h)

        # Middle
        h = self.mid_res1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, temb)

        # Decoder
        for level, up in zip(self.up_blocks, self.up_samples):
            for blk in level:
                h = torch.cat([h, skips.pop()], dim=1)
                h = blk[0](h, temb)
                if len(blk) > 1:
                    h = blk[1](h)
            if up is not None:
                h = up(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


if __name__ == "__main__":
    net = UNet(img_size=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = net(x, t)
    n_param = sum(p.numel() for p in net.parameters())
    print(f"UNet: x{tuple(x.shape)}, t{tuple(t.shape)} -> {tuple(y.shape)}")
    print(f"params: {n_param/1e6:.2f}M")
