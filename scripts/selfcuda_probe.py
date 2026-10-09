"""抓 torch.profiler，按 Self CUDA 排序打印 Top 叶子算子，用于报告演示。

与 ncu 的区别：torch.profiler 按 aten 算子/kernel 名聚合，能看到 kernel 名
(如 sm80_xmma_fprop...)，但看不到 HMMA/FMA 等硬件 pipe 计数器(那只有 ncu 能读)。

用法：
    python scripts/selfcuda_probe.py --dtype bf16 --top 8
"""

import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, schedule

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unet"))

from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "tf32", "bf16"], default="bf16")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--top", type=int, default=4)
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

    sched = schedule(wait=1, warmup=1, active=args.steps, repeat=1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched, record_shapes=False, profile_memory=False) as prof:
        for _ in range(2 + args.steps):
            with torch.no_grad():
                net(x, t)
            torch.cuda.synchronize()
            prof.step()

    evts = prof.key_averages()

    def self_cuda(e):
        v = getattr(e, "self_device_time_total", None)
        return v if v is not None else e.self_cuda_time_total

    # 只保留真正吃 GPU 的叶子算子/kernel；容器层(ProfilerStep*/forward/各 module range)
    # 的 Self CUDA 是包裹语义，不是独立 kernel，这里按名字过滤掉。
    def is_leaf(e):
        return e.key.startswith("aten::") or e.key.startswith("void")

    leaves = [e for e in evts if is_leaf(e)]
    top = sorted(leaves, key=self_cuda, reverse=True)[:args.top]

    total = sum(self_cuda(e) for e in leaves)
    print(f"\n=== dtype={args.dtype} batch={args.batch} size={args.size} "
          f"steps={args.steps} ===")
    print(f"叶子算子 Self CUDA 合计 = {total / 1e3:.3f} ms\n")
    print(f"{'算子':<40}{'Self CUDA(ms)':>14}{'占比':>9}{'调用数':>8}")
    print("-" * 71)
    for e in top:
        sc = self_cuda(e) / 1e3
        name = e.key if len(e.key) <= 38 else e.key[:35] + "..."
        print(f"{name:<40}{sc:>14.3f}{100 * self_cuda(e) / total:>8.1f}%{e.count:>8}")


if __name__ == "__main__":
    main()
