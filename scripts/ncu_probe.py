"""ncu 专用最小驱动：预热后只在 cudaProfiler 范围内跑 1 次前向，供 Nsight Compute 抓取。

配合 scripts/run_ncu.sh 使用（它会在 gcc-12 loader 下用 ncu 包起本脚本）：
    sudo ncu --profile-from-start off --target-processes all \\
        --metrics ... bash scripts/run_ncu.sh probe --dtype fp32

精度口径与 unet/benchmark.py 保持一致：
    fp32  关闭 TF32，强制走真正的 FP32（winograd / CUDA core）
    tf32  打开 TF32 Tensor Core 路径
    bf16  权重与浮点输入一次性转 bf16，全程不走 autocast（部署态）
"""

import argparse
import os
import sys

import torch

# scripts/ 与 unet/ 平级，把 unet 目录加入 import 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unet"))

from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "tf32", "bf16"], default="fp32")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    use_tf32 = args.dtype == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    dev = torch.device("cuda")
    net = UNet(img_size=args.size).to(dev).eval()
    x = torch.randn(args.batch, 3, args.size, args.size, device=dev)
    t = torch.randint(0, 1000, (args.batch,), device=dev)

    # 纯 bf16 部署态：权重与浮点输入一次性转 bf16，全程不走 autocast。
    if args.dtype == "bf16":
        net = net.to(torch.bfloat16)
        x = x.to(torch.bfloat16)

    for _ in range(args.warmup):
        with torch.no_grad():
            net(x, t)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    with torch.no_grad():
        net(x, t)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
