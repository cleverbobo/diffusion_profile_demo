#!/bin/bash
# ncu 采集驱动：在 gcc-12 自定义 loader 下运行 ncu_probe.py，并封装 Nsight Compute
# 的采集命令。本机驱动 RmProfilingAdminOnly=1，读硬件计数器需 root，故用免密 sudo。
#
# 用法：
#   bash scripts/run_ncu.sh probe --dtype fp32          # 仅在 loader 下跑一次前向(供 ncu 包裹)
#   bash scripts/run_ncu.sh full  fp32                  # 采全部 kernel  -> /tmp/ncu_fp32.ncu-rep
#   bash scripts/run_ncu.sh conv  bf16                  # 只采卷积 kernel -> /tmp/ncu_conv_bf16.ncu-rep
#
# 采完后导出 CSV（import 不需 sudo）再用 ncu_report.py / ncu_conv_roofline.py 分析：
#   ncu --import /tmp/ncu_fp32.ncu-rep --csv --page raw > /tmp/ncu_fp32.csv
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ENV=/home/disk3/wanhaibo/digital_human/haibo_work/env/anaconda3/envs/fastvideo
LOADER=/opt/compiler/gcc-12/lib64/ld-linux-x86-64.so.2
NV=$(ls -d "$ENV"/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
NCU_TGT=/usr/local/cuda/nsight-compute-2022.4.0/target/linux-desktop-glibc_2_11_3-x64

# 关键：/usr/lib64 必须排在 gcc-12/lib64 之后，否则触发 _dl_starting_up GLIBC_PRIVATE 冲突
run_probe() {
  exec "$LOADER" \
    --library-path "/opt/compiler/gcc-12/lib64:$NV:/lib64:/usr/lib64:$NCU_TGT" \
    "$ENV/bin/python" "$HERE/ncu_probe.py" "$@"
}

# roofline / pipe 判定需要的 5 个计数器
METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"

# 卷积 kernel 名正则（focused capture 只插桩这些）
CONV_RE="fprop|winograd|scudnn|sgemm|gemm|cutlass|convolve"

capture() {  # $1=输出名 $2=dtype  [$3=卷积正则]
  local out=$1 dtype=$2 kfilter=${3:-}
  local extra=""
  [ -n "$kfilter" ] && extra="--kernel-name regex:$kfilter"
  sudo ncu --profile-from-start off --target-processes all \
    --metrics "$METRICS" $extra \
    -f -o "/tmp/$out" \
    bash "$HERE/run_ncu.sh" probe --dtype "$dtype"
}

cmd=${1:-}
shift || true
case "$cmd" in
  probe) run_probe "$@" ;;
  full)  capture "ncu_${1}" "$1" ;;
  conv)  capture "ncu_conv_${1}" "$1" "$CONV_RE" ;;
  *) echo "用法: $0 {probe --dtype X | full {fp32|tf32|bf16} | conv {fp32|tf32|bf16}}" >&2; exit 1 ;;
esac
