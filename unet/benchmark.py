"""扩散 UNet 性能测试

测量: 计算量(FLOPs/MACs)、前向/反向延迟、吞吐(imgs/s)、实际算力(TFLOPS)、
参数量、峰值显存(GPU)。自动适配 CPU/GPU。

计算量用 torch.utils.flop_counter.FlopCounterMode 统计，只计 conv/matmul/attention
等矩阵类算子(占绝大部分)，GroupNorm、激活、加法等逐元素算子不计入。
1 MAC = 2 FLOPs。

算力(TFLOPS)依赖计算类型(精度)：fp32 / tf32 / bf16 的峰值算力差异巨大
(例如同一张 GPU 上 tf32/bf16 的峰值常是 fp32 的数倍)，因此所有 TFLOPS 结果都会
标注对应的计算类型，脱离计算类型谈算力没有意义。用 --dtype 选择：
    fp32  纯 FP32，matmul 走 FP32 (基准)
    tf32  FP32 存储，matmul/conv 走 TF32 (仅 Ampere+ GPU)
    bf16  纯 BF16，权重与激活一次性转 bf16 (无 autocast，部署态)

用法:
    python benchmark.py                         # 默认测完整 UNet 并生成 trace
    python benchmark.py --dtype tf32            # 用 TF32 计算类型测算力
    python benchmark.py --dtype bf16 --backward # 纯 BF16，含反向
    python benchmark.py --target unet --backward --batch 16 --size 32
    python benchmark.py --target res            # 只测 ResidualBlock
    python benchmark.py --target attn --size 32 --in-ch 256
    python benchmark.py --no-profile            # 只跑 benchmark，不抓 trace
    python benchmark.py --backward --trace-dir ./traces

trace 文件为 Chrome trace 格式(.json)，可用 https://ui.perfetto.dev 或
chrome://tracing 打开；trace 中每个 nn.Module 都有以模块名命名的区间。
"""

import argparse
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.flop_counter import FlopCounterMode

from blocks import AttentionBlock, DownSample, ResidualBlock, UpSample
from unet import UNet


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def setup_precision(dtype, device):
    """按计算类型配置后端。

    - tf32 : 打开 cuBLAS/cuDNN 的 TF32 路径(仅 Ampere+ GPU 生效)
    - fp32 : 关闭 TF32，matmul/conv 走纯 FP32
    - bf16 : 关闭 TF32；权重/激活在 build 后一次性转 bf16(见 bench)，不走 autocast
    """
    use_tf32 = dtype == "tf32"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32


def count_flops(net, run, backward=False):
    """统计一次 run() 的 FLOPs；backward=True 时统计前向+反向总量。"""
    with FlopCounterMode(display=False) as fc:
        if backward:
            net.zero_grad(set_to_none=True)
            out = run()
            out.sum().backward()
        else:
            with torch.no_grad():
                run()
    return fc.get_total_flops()


def _fmt_flops(f):
    return f"{f/1e9:10.3f} GFLOPs"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def build(target, device, b, s, in_ch):
    """返回 (net, inputs)。inputs 为前向入参元组，run() 即 net(*inputs)。"""
    temb_ch = 256
    if target == "unet":
        net = UNet(img_size=s).to(device)
        x = torch.randn(b, 3, s, s, device=device)
        t = torch.randint(0, 1000, (b,), device=device)
        return net, (x, t)
    if target == "res":
        net = ResidualBlock(in_ch, in_ch, temb_ch).to(device)
        x = torch.randn(b, in_ch, s, s, device=device)
        temb = torch.randn(b, temb_ch, device=device)
        return net, (x, temb)
    if target == "attn":
        net = AttentionBlock(in_ch).to(device)
        x = torch.randn(b, in_ch, s, s, device=device)
        return net, (x,)
    if target == "down":
        net = DownSample(in_ch).to(device)
        x = torch.randn(b, in_ch, s, s, device=device)
        return net, (x,)
    if target == "up":
        net = UpSample(in_ch).to(device)
        x = torch.randn(b, in_ch, s, s, device=device)
        return net, (x,)
    raise ValueError(target)


def bench(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}")

    setup_precision(args.dtype, device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"compute dtype={args.dtype}  ({gpu})")

    net, inputs = build(args.target, device, args.batch, args.size, args.in_ch)
    # bf16 = 纯 bf16 部署模式：把权重和浮点输入一次性转成 bf16，全程不走 autocast。
    # 整数输入(如时间步 t)保持原 dtype。
    if args.dtype == "bf16":
        net = net.to(torch.bfloat16)
        inputs = tuple(
            v.to(torch.bfloat16) if torch.is_floating_point(v) else v for v in inputs
        )
    run = lambda: net(*inputs)
    print(f"target={args.target}  params={count_params(net)/1e6:.3f}M  "
          f"batch={args.batch}  size={args.size}")

    # 计算量(与设备无关，只统计一次)。在 eager 下统计，避免 torch.compile
    # 的图执行绕过 FlopCounterMode 的 dispatch 导致计不到算子。
    fwd_flops = count_flops(net, run)
    bwd_flops = count_flops(net, run, backward=True) if args.backward else None

    # torch.compile（默认关闭，--torch_compile 开启）。编译后重建 run，
    # 编译开销由后续 warmup 吸收。
    if args.torch_compile:
        print(f"torch.compile enabled (mode={args.compile_mode})")
        net = torch.compile(net, mode=args.compile_mode)
        run = lambda: net(*inputs)

    # 预热。compile 模式下编译在此完成：分别预热前向(no_grad)与反向(grad)路径，
    # 否则 autograd 开关变化会触发 torch.compile 重新编译，拖慢计时循环。
    for _ in range(args.warmup):
        with torch.no_grad():
            run()
        if args.backward:
            net.zero_grad(set_to_none=True)
            out = run()
            out.sum().backward()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 前向
    t0 = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            run()
    _sync(device)
    fwd_ms = (time.perf_counter() - t0) / args.iters * 1e3

    # 前向 + 反向
    bwd_ms = None
    if args.backward:
        t0 = time.perf_counter()
        for _ in range(args.iters):
            net.zero_grad(set_to_none=True)
            out = run()
            out.sum().backward()
        _sync(device)
        bwd_ms = (time.perf_counter() - t0) / args.iters * 1e3

    b = args.batch
    thpt = b / (fwd_ms / 1e3)
    dt = args.dtype
    print("-" * 64)
    print("[计算量]")
    print(f"forward     : {_fmt_flops(fwd_flops)}  (每张 {fwd_flops/b/1e9:.3f} GFLOPs"
          f" / {fwd_flops/b/2/1e9:.3f} GMACs)")
    if bwd_flops is not None:
        print(f"fwd+backward: {_fmt_flops(bwd_flops)}  (约为前向的 {bwd_flops/fwd_flops:.2f} 倍)")
    print(f"[性能]  (算力按计算类型 {dt} 计)")
    print(f"forward     : {fwd_ms:8.3f} ms/iter   ({thpt:8.1f} imgs/s)"
          f"   实际算力 {fwd_flops/(fwd_ms/1e3)/1e12:.3f} TFLOPS[{dt}]")
    if bwd_ms is not None:
        print(f"fwd+backward: {bwd_ms:8.3f} ms/iter"
              f"   实际算力 {bwd_flops/(bwd_ms/1e3)/1e12:.3f} TFLOPS[{dt}]")
    if device.type == "cuda":
        print(f"peak memory : {torch.cuda.max_memory_allocated()/1024**2:8.1f} MB")

    if args.profile:
        run_profile(args, net, run, device)
    else:
        print("-" * 64)
        print("已跳过 trace 抓取(--no-profile)")


def _add_module_ranges(net):
    """给每个子模块挂 hook，使 trace 中出现以模块名命名的区间(如 down_blocks.0.1.0)。"""
    handles = []
    for name, m in net.named_modules():
        label = name or type(net).__name__

        def pre(mod, _inp, _label=label):
            rf = record_function(_label)
            rf.__enter__()
            mod._prof_rf = getattr(mod, "_prof_rf", []) + [rf]

        def post(mod, _inp, _out):
            mod._prof_rf.pop().__exit__(None, None, None)

        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post))
    return handles


def run_profile(args, net, run, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(args.trace_dir, exist_ok=True)
    mode = "train" if args.backward else "fwd"
    trace_path = os.path.join(
        args.trace_dir,
        f"{args.target}_b{args.batch}_s{args.size}_{args.dtype}_{mode}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )

    handles = _add_module_ranges(net)
    # wait=1 跳过首步, warmup=1 让 profiler 自身开销稳定, active=N 步真正记录
    sched = schedule(wait=1, warmup=1, active=args.profile_steps, repeat=1)
    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    ) as prof:
        for _ in range(2 + args.profile_steps):
            if args.backward:
                net.zero_grad(set_to_none=True)
                with record_function("forward"):
                    out = run()
                with record_function("backward"):
                    out.sum().backward()
            else:
                with torch.no_grad(), record_function("forward"):
                    run()
            _sync(device)
            prof.step()
    for h in handles:
        h.remove()

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("-" * 64)
    print(f"[Profiler] 记录 {args.profile_steps} 步，按 {sort_key} 排序的算子 Top {args.profile_top}:")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.profile_top))
    print(f"trace 已保存: {os.path.abspath(trace_path)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["unet", "res", "attn", "down", "up"],
                   default="unet")
    p.add_argument("--in-ch", type=int, default=128, dest="in_ch")
    p.add_argument("--dtype", choices=["fp32", "tf32", "bf16"],
                   default="fp32", help="计算类型(精度)，决定算力口径；bf16=纯bf16(无autocast)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--torch_compile", action="store_true", dest="torch_compile",
                   help="用 torch.compile 编译模型(默认关闭)")
    p.add_argument("--compile-mode", default="default", dest="compile_mode",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile 的 mode，仅在 --torch_compile 时生效")
    p.add_argument("--no-profile", action="store_false", dest="profile",
                   help="只跑 benchmark，不用 torch.profiler 抓 trace")
    p.set_defaults(profile=True)
    p.add_argument("--profile-steps", type=int, default=3, dest="profile_steps",
                   help="profiler 实际记录的步数")
    p.add_argument("--profile-top", type=int, default=20, dest="profile_top",
                   help="终端打印的算子条数")
    p.add_argument("--trace-dir", default="./traces", dest="trace_dir")
    bench(p.parse_args())


if __name__ == "__main__":
    main()
