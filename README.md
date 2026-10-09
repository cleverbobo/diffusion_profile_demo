# 扩散 UNet 性能测试 demo

DDPM 风格扩散模型 UNet 的 PyTorch 实现，附带一套性能分析工具：torch 层面的
延迟 / 吞吐 / FLOPs / 显存 + trace，以及 Nsight Compute (ncu) 核级的 Tensor Core /
CUDA Core / 带宽利用率分析。实测硬件为 **NVIDIA A10**（Ampere，24GB，600 GB/s）。

完整的测试结论见 [`benchmark_report.md`](benchmark_report.md)。

## 目录结构

```
diffusion_profile_demo/
├── README.md               本文件
├── requirements.txt        依赖(仅 torch)
├── env.sh                  conda 环境 + gcc-12 自定义 loader + torch.compile 环境
├── benchmark_report.md     性能测试报告(fp32/tf32/bf16 横向对比 + ncu 核级分析)
├── images/unet.png         结构图
├── unet/                   模型 + 主 benchmark
│   ├── blocks.py           核心 block(时间步编码/残差/注意力/上下采样)
│   ├── unet.py             用 blocks 堆叠出的完整 UNet
│   └── benchmark.py        延迟/吞吐/FLOPs/算力/显存 + torch.profiler trace
└── scripts/                profiling / ncu 分析脚本
    ├── ncu_probe.py        ncu 用的最小前向驱动(只在 cudaProfiler 范围内跑 1 次)
    ├── run_ncu.sh          gcc-12 loader 包装 + ncu 采集命令(full / conv)
    ├── ncu_report.py       ncu CSV 按算子类别聚合(conv/norm/elementwise/layout)
    ├── ncu_conv_roofline.py focused 卷积 roofline/瓶颈分析(SM/DRAM/HMMA/FMA)
    └── selfcuda_probe.py   torch.profiler 按 Self CUDA 排序打印 Top 算子
```

## 环境准备

```bash
pip install -r requirements.txt      # 仅 torch(实测 2.10.0+cu128)
```

本仓库运行在 conda `fastvideo` 环境 + gcc-12 自定义 loader 下，`env.sh` 封装了
环境激活、`python_env` alias 以及 torch.compile 所需的 `CC/CXX/CUDA` 变量：

```bash
source env.sh
```

> 注意：自定义 loader 的 `--library-path` 里 `/usr/lib64` 必须排在 `gcc-12/lib64`
> 之后，否则会触发 `_dl_starting_up` 的 `GLIBC_PRIVATE` 冲突。

## 模型结构（`unet/`）

`blocks.py` 提供各图元，`unet.py` 把它们堆成完整 UNet：

| 类 | 作用 |
|----|------|
| `TimeEmbedding` | 正弦位置编码 + MLP，把时间步 `t` 编码为向量注入每个残差块 |
| `ResidualBlock` | GroupNorm-SiLU-Conv ×2 + 时间步注入 + 残差 |
| `AttentionBlock` | GroupNorm + 空间自注意力(用 `F.scaled_dot_product_attention`) + 残差 |
| `DownSample` | stride=2 卷积下采样 |
| `UpSample` | 最近邻插值 + 卷积上采样 |

前向流程：

```
Conv(in_ch -> base_ch)
├ Encoder: 每个分辨率 level 堆 num_res_blocks 个 ResidualBlock(+Attn)，保存 skip
│          非最后一层接 DownSample
├ Middle : ResidualBlock -> AttentionBlock -> ResidualBlock
└ Decoder: 每个 level 堆 num_res_blocks+1 个 (concat skip -> ResidualBlock(+Attn))
           非最后一层接 UpSample
GroupNorm-SiLU-Conv(base_ch -> out_ch)
```

默认配置：`base_ch=64`，`ch_mult=(1,2,4,8)`，`num_res_blocks=2`，`attn_resolutions=(8,)`，
`groups=32`。注意力只在特征图降到 **8×8** 时插入——所以 `size=512`（3 次下采样后最小
64×64，到不了 8）下**编码/解码器都没有注意力**，只有中间块一个固定注意力；而 `size=32`
会在 8×8 处命中注意力，参数量也因此略有不同（512 下约 **56.6M**，32 下约 **57.9M**）。

### 形状自检

```bash
cd unet
python blocks.py   # 打印各 block 的输入输出形状
python unet.py     # 打印完整 UNet 的输入输出与参数量
```

## 性能测试（`unet/benchmark.py`）

统计计算量(FLOPs/MACs)、前向/反向延迟、吞吐、实际算力(TFLOPS)、参数量、峰值显存，
并可选抓 torch.profiler trace。自动适配 CPU/GPU。

```bash
cd unet
# 报告口径：batch=1, size=512，三档精度
python benchmark.py --dtype fp32 --batch 1 --size 512   # 纯 FP32(CUDA core, winograd)
python benchmark.py --dtype tf32 --batch 1 --size 512   # TF32 Tensor Core
python benchmark.py --dtype bf16 --batch 1 --size 512   # 纯 bf16 部署态(无 autocast)

python benchmark.py --no-profile                        # 只跑 benchmark，不抓 trace
python benchmark.py --backward                          # 同时测反向
python benchmark.py --target res --in-ch 128 --size 32  # 只测单个 block
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--target` | `unet` | `unet` / `res` / `attn` / `down` / `up` |
| `--dtype` | `fp32` | 计算类型(算力口径)：`fp32` / `tf32` / `bf16`(纯 bf16，无 autocast) |
| `--batch` | 1 | batch size |
| `--size` | 32 | 输入边长(报告用 512) |
| `--in-ch` | 128 | 单 block 测试的输入通道 |
| `--iters` / `--warmup` | 10 / 5 | 计时 / 预热迭代数 |
| `--backward` | 关 | 是否同时测反向 |
| `--torch_compile` | 关 | 用 torch.compile 编译(配 `--compile-mode`) |
| `--no-profile` | — | 关闭 torch.profiler trace(默认开) |
| `--profile-steps` / `--profile-top` | 3 / 20 | profiler 记录步数 / 打印 Top N |
| `--trace-dir` | `./traces` | trace 输出目录 |

> FLOPs 只统计 conv/matmul/attention 等矩阵类算子(1 MAC = 2 FLOPs)，GroupNorm/激活/
> 逐元素算子不计入；它们是 memory-bound，耗时体现在 MFU 的固定开销里。
> trace 为 Chrome trace(.json)，用 https://ui.perfetto.dev 或 `chrome://tracing` 打开，
> 每个 `nn.Module` 都有以模块名命名的区间，可直接定位到哪个 block 慢。

## ncu 核级分析（`scripts/`）

torch.profiler 能看到 kernel 名，但读不到硬件 pipe 计数器；判定某个 kernel 跑在
Tensor Core(HMMA) 还是 CUDA Core(FMA)、带宽(DRAM)是否打满，需要 Nsight Compute。
本机驱动 `RmProfilingAdminOnly=1`，读计数器需 root，`run_ncu.sh` 用免密 `sudo` 包装。

```bash
# 1) 采集(ncu replay 每个 kernel，较慢)。full=全部 kernel；conv=只插桩卷积 kernel
bash scripts/run_ncu.sh full fp32      # -> /tmp/ncu_fp32.ncu-rep
bash scripts/run_ncu.sh conv bf16      # -> /tmp/ncu_conv_bf16.ncu-rep

# 2) 导出 CSV(import 不需 sudo)
ncu --import /tmp/ncu_fp32.ncu-rep --csv --page raw > /tmp/ncu_fp32.csv
ncu --import /tmp/ncu_conv_bf16.ncu-rep --csv --page raw > /tmp/ncu_conv_bf16.csv

# 3) 分析
python scripts/ncu_report.py /tmp/ncu_fp32.csv:fp32            # 按算子类别聚合
python scripts/ncu_conv_roofline.py /tmp/ncu_conv_bf16.csv:bf16  # 卷积 roofline/瓶颈

# torch.profiler Self CUDA Top 算子(对照用，无需 sudo/ncu)
python scripts/selfcuda_probe.py --dtype bf16 --top 8
```

判据：`HMMA% > 0` → 用了 Tensor Core；`FMA% > 0` → 用了 CUDA Core；`SM%` 为计算 SOL、
`DRAM%` 为带宽 SOL，`DRAM ≥ SM` 判带宽 bound，否则偏计算侧(SM 接近峰值才算打满)。
