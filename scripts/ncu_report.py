"""聚合 ncu `--csv --page raw` 输出：按算子类别(conv/groupnorm/elementwise/layout/
concat/other)统计耗时与 SM%/DRAM%/HMMA%/FMA%(按耗时加权)，并列出卷积 Top5。

列按表头名解析，不依赖 --metrics 顺序。ncu raw CSV 第 0 行是表头、第 1 行是单位行，
数据从第 2 行开始。

用法：
    ncu --import /tmp/ncu_fp32.ncu-rep --csv --page raw > /tmp/ncu_fp32.csv
    python scripts/ncu_report.py /tmp/ncu_fp32.csv:fp32 /tmp/ncu_bf16.csv:bf16
"""

import csv
import sys


NAME_COL = "Kernel Name"
DUR_COL = "gpu__time_duration.sum"
SM_COL = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_COL = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
HMMA_COL = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
FMA_COL = "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"


def classify(name):
    """按 kernel 名归类。注意 layout 检查必须在 conv 之前(nchwToNhwc 名里含 cudnn)。"""
    n = name.lower()
    if "nchwtonhwc" in n or "nhwctonchw" in n or "layout" in n:
        return "layout"
    if "groupnorm" in n or "group_norm" in n or "rowwisemoments" in n or \
       "welford" in n or "normalization" in n:
        return "groupnorm"
    if "concat" in n:
        return "concat"
    if any(k in n for k in ["sgemm", "fprop", "implicit_gemm", "winograd", "conv",
                            "xmma", "cudnn", "wgrad", "dgrad", "cutlass", "gemm"]):
        return "conv"
    if "elementwise" in n or "silu" in n or "add" in n or "vectorized" in n \
       or "unrolled" in n:
        return "elementwise"
    return "other"


def col_index(header, name):
    """在表头里按名字(容忍前缀匹配)找列号。"""
    for i, h in enumerate(header):
        if h.strip() == name:
            return i
    for i, h in enumerate(header):
        if h.strip().startswith(name):
            return i
    raise KeyError(f"列未找到: {name}")


def load(path):
    rows = list(csv.reader(open(path)))
    header = rows[0]
    ci = {k: col_index(header, k) for k in
          [NAME_COL, DUR_COL, SM_COL, DRAM_COL, HMMA_COL, FMA_COL]}
    cats = {}
    kernels = {}
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
        name = row[ci[NAME_COL]]
        c = classify(name)
        d = cats.setdefault(c, [0.0, 0.0, 0.0, 0.0, 0.0])
        d[0] += dur
        d[1] += dur * sm
        d[2] += dur * dram
        d[3] += dur * hmma
        d[4] += dur * fma
        k = kernels.setdefault(name, [0.0, 0.0, 0.0, c])
        k[0] += dur
        k[1] += dur * hmma
        k[2] += dur * fma
    return cats, kernels


def report(path, label):
    cats, kernels = load(path)
    total = sum(v[0] for v in cats.values())
    print(f"\n===== {label}  GPU合计={total:.3f} ms =====")
    print(f"  {'类别':<12}{'ms':>9}{'占比':>7}{'SM%':>7}{'DRAM%':>7}{'HMMA%':>7}{'FMA%':>7}")
    order = ["conv", "groupnorm", "elementwise", "layout", "concat", "other"]
    for c in order:
        if c not in cats:
            continue
        t, wsm, wdram, wh, wf = cats[c]
        print(f"  {c:<12}{t:>9.3f}{100 * t / total:>6.1f}%"
              f"{wsm / t:>7.1f}{wdram / t:>7.1f}{wh / t:>7.1f}{wf / t:>7.1f}")
    print("  -- conv top5 (按耗时) --")
    convs = sorted(([k] + v for k, v in kernels.items() if v[3] == "conv"),
                   key=lambda x: x[1], reverse=True)[:5]
    for name, t, wh, wf, _c in convs:
        print(f"     {name[:46]:<46}{t:>8.3f} ms  HMMA={wh / t:>5.1f}  FMA={wf / t:>5.1f}")


def main():
    for arg in sys.argv[1:]:
        path, label = arg.split(":")
        report(path, label)


if __name__ == "__main__":
    main()
