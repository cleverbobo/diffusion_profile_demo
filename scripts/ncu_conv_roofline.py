"""分析 ncu 卷积 focused capture：对每类卷积 kernel 统计耗时、SM(计算)SOL%、
DRAM(带宽)SOL%、HMMA(Tensor Core)%、FMA(CUDA Core)%，并给出瓶颈判定。

列按表头名解析，不依赖 --metrics 顺序。瓶颈判据：
    max(SM, DRAM) < 55   -> 延迟/占用(未饱和)
    DRAM >= SM           -> 带宽 bound
    否则                 -> 计算侧(SM>DRAM；是否打满看 SM 是否接近峰值)

用法：
    ncu --import /tmp/ncu_conv_bf16.ncu-rep --csv --page raw > /tmp/ncu_conv_bf16.csv
    python scripts/ncu_conv_roofline.py /tmp/ncu_conv_fp32.csv:fp32 \\
        /tmp/ncu_conv_bf16.csv:bf16
"""

import csv
import sys


NAME_COL = "Kernel Name"
DUR_COL = "gpu__time_duration.sum"
SM_COL = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_COL = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
HMMA_COL = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
FMA_COL = "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"


def group(name):
    """把卷积 kernel 归成可辨识的组，括号标注预期走 TC(Tensor Core)还是 CC(CUDA Core)。"""
    n = name.lower()
    if "xmma_fprop" in n and "indexed" in n:
        return "xmma_fprop_indexed(TC)"
    if "xmma_fprop" in n:
        return "xmma_fprop(TC主力)"
    if "winograd" in n and "generate" in n:
        return "winograd_tiles(辅助)"
    if "winograd" in n:
        return "scudnn_winograd(CC)"
    if "scudnn" in n:
        return "scudnn_small(CC)"
    if "tensorop" in n:
        return "cutlass_tensorop(TC)"
    if "simt_sgemm" in n or "sgemm_128x64_tn" in n or \
       ("sgemm" in n and "bf16" not in n and "1688" not in n and "16816" not in n):
        return "sgemm_simt(CC)"
    if "s1688gemm" in n or "s16816gemm" in n or "1688gemm" in n or "16816gemm" in n:
        return "sgemm_bf16(TC)"
    if "convolve_sgemm" in n:
        return "implicit_convolve(CC)"
    return "other_conv"


def col_index(header, name):
    for i, h in enumerate(header):
        if h.strip() == name:
            return i
    for i, h in enumerate(header):
        if h.strip().startswith(name):
            return i
    raise KeyError(f"列未找到: {name}")


def run(path, label):
    rows = list(csv.reader(open(path)))
    header = rows[0]
    ci = {k: col_index(header, k) for k in
          [NAME_COL, DUR_COL, SM_COL, DRAM_COL, HMMA_COL, FMA_COL]}
    agg = {}
    for row in rows[2:]:  # 跳过表头 + 单位行
        if len(row) <= max(ci.values()):
            continue
        try:
            dur = float(row[ci[DUR_COL]])
            sm = float(row[ci[SM_COL]])
            dram = float(row[ci[DRAM_COL]])
            hmma = float(row[ci[HMMA_COL]])
            fma = float(row[ci[FMA_COL]])
        except ValueError:
            continue
        g = group(row[ci[NAME_COL]])
        a = agg.setdefault(g, [0.0, 0, 0.0, 0.0, 0.0, 0.0])
        a[0] += dur
        a[1] += 1
        a[2] += dur * sm
        a[3] += dur * dram
        a[4] += dur * hmma
        a[5] += dur * fma
    total = sum(a[0] for a in agg.values())
    tc_time = sum(a[0] for g, a in agg.items() if "(TC" in g)
    cc_time = sum(a[0] for g, a in agg.items() if "(CC)" in g or "convolve" in g)
    print(f"\n========== {label}  卷积总耗时={total:.3f} ms ==========")
    print(f"{'卷积 kernel 组':<26}{'ms':>8}{'占比':>7}{'calls':>6}"
          f"{'SM%':>7}{'DRAM%':>7}{'HMMA%':>7}{'FMA%':>7}  瓶颈")
    print("-" * 94)
    for g, a in sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True):
        t, c, wsm, wdram, wh, wf = a
        sm, dram, hmma, fma = wsm / t, wdram / t, wh / t, wf / t
        if max(sm, dram) < 55:
            bn = "延迟/占用"
        elif dram >= sm:
            bn = "带宽bound"
        else:
            bn = "计算侧" if sm < 78 else "计算bound"
        print(f"{g:<26}{t:>8.3f}{100 * t / total:>6.1f}%{c:>6}"
              f"{sm:>7.1f}{dram:>7.1f}{hmma:>7.1f}{fma:>7.1f}  {bn}")
    print(f"  Tensor Core 路径卷积 = {tc_time:.3f} ms ({100 * tc_time / total:.1f}%)"
          f" | CUDA Core 路径卷积 = {cc_time:.3f} ms ({100 * cc_time / total:.1f}%)")


def main():
    for arg in sys.argv[1:]:
        path, label = arg.split(":")
        run(path, label)


if __name__ == "__main__":
    main()
