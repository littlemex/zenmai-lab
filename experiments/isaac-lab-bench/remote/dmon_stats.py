#!/usr/bin/env python3
"""Aggregate `nvidia-smi dmon -s upcm -d 1 -o DT` logs.

Reads one or more dmon log files (or globs) and prints per-file stats:
SM%, MEM-bandwidth%, power, memory & graphics clock — averages, p95, max.

Examples
--------
  # Single file
  python dmon_stats.py path/to/nvidia-smi-dmon.log

  # Glob (works on the EC2 side too, since shells expand the glob)
  python dmon_stats.py /mnt/s3files/bench/g1-prod/run-seed42-*/nvidia-smi-dmon.log

  # Markdown table output for pasting into reports
  python dmon_stats.py --format md run-*/nvidia-smi-dmon.log
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
from dataclasses import dataclass


# nvidia-smi dmon -s upcm -d 1 -o DT layout (post-driver R535+):
#   col 0   Date (YYYYMMDD)
#   col 1   Time (HH:MM:SS)
#   col 2   gpu (Idx)
#   col 3   sm   (%)
#   col 4   mem  (%)
#   col 5   enc  (%)
#   col 6   dec  (%)
#   col 7   jpg  (%)
#   col 8   ofa  (%)
#   col 9   pwr  (W)
#   col 10  gtemp
#   col 11  mtemp
#   col 12  mclk (MHz)
#   col 13  pclk (MHz)
#   col 14  fb   (MB)
#   col 15  bar1 (MB)
#   col 16  ccpm (MB)
#
# Lines that are headers/comments start with '#'. Data lines start with the
# date (a leading number). We accept any digit-leading row to stay
# year-agnostic.


@dataclass
class Stats:
    name: str
    n: int
    sm_avg: float
    sm_p95: int
    sm_max: int
    mem_avg: float
    mem_max: int
    pwr_avg: float
    pwr_max: int
    mclk_avg: float
    pclk_avg: float


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * p), len(sorted_v) - 1)
    return sorted_v[idx]


def parse_file(path: str) -> Stats | None:
    sm: list[int] = []
    mem: list[int] = []
    pwr: list[int] = []
    mclk: list[int] = []
    pclk: list[int] = []
    with open(path) as fp:
        for line in fp:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            # data row begins with the date (digits)
            if not stripped[0].isdigit():
                continue
            parts = stripped.split()
            if len(parts) < 14:
                continue
            try:
                sm.append(int(parts[3]))
                mem.append(int(parts[4]))
                pwr.append(int(parts[9]))
                mclk.append(int(parts[12]))
                pclk.append(int(parts[13]))
            except (ValueError, IndexError):
                continue

    if not sm:
        return None

    # The directory name of the run is more useful than the bare filename.
    parent = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    return Stats(
        name=parent,
        n=len(sm),
        sm_avg=statistics.mean(sm),
        sm_p95=percentile(sm, 0.95),
        sm_max=max(sm),
        mem_avg=statistics.mean(mem),
        mem_max=max(mem),
        pwr_avg=statistics.mean(pwr),
        pwr_max=max(pwr),
        mclk_avg=statistics.mean(mclk),
        pclk_avg=statistics.mean(pclk),
    )


def emit_text(stats: list[Stats]) -> None:
    for s in stats:
        print(f"=== {s.name} ===")
        print(
            f"  n={s.n} sm_avg={s.sm_avg:.1f}% sm_p95={s.sm_p95}% sm_max={s.sm_max}%"
        )
        print(f"  mem_avg={s.mem_avg:.1f}% mem_max={s.mem_max}%")
        print(f"  pwr_avg={s.pwr_avg:.0f}W pwr_max={s.pwr_max}W")
        print(f"  mclk_avg={s.mclk_avg:.0f}MHz pclk_avg={s.pclk_avg:.0f}MHz")


def emit_csv(stats: list[Stats]) -> None:
    print(
        "name,samples,sm_avg,sm_p95,sm_max,mem_avg,mem_max,"
        "pwr_avg,pwr_max,mclk_avg,pclk_avg"
    )
    for s in stats:
        print(
            f"{s.name},{s.n},{s.sm_avg:.1f},{s.sm_p95},{s.sm_max},"
            f"{s.mem_avg:.1f},{s.mem_max},{s.pwr_avg:.0f},{s.pwr_max},"
            f"{s.mclk_avg:.0f},{s.pclk_avg:.0f}"
        )


def emit_md(stats: list[Stats]) -> None:
    print(
        "| run | n | SM avg | SM p95 | SM max | MEM avg | MEM max | "
        "PWR avg | PWR max | mclk | pclk |"
    )
    print(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for s in stats:
        print(
            f"| {s.name} | {s.n} | {s.sm_avg:.1f}% | {s.sm_p95}% | {s.sm_max}% "
            f"| {s.mem_avg:.1f}% | {s.mem_max}% | {s.pwr_avg:.0f} W | "
            f"{s.pwr_max} W | {s.mclk_avg:.0f} MHz | {s.pclk_avg:.0f} MHz |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="dmon log files or globs")
    parser.add_argument(
        "--format", choices=("text", "csv", "md"), default="text"
    )
    args = parser.parse_args()

    files: list[str] = []
    for p in args.paths:
        if any(c in p for c in "*?["):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)

    if not files:
        print("no files matched", file=sys.stderr)
        return 1

    stats = [s for s in (parse_file(f) for f in files) if s is not None]
    if not stats:
        print("no usable data points found", file=sys.stderr)
        return 1

    {"text": emit_text, "csv": emit_csv, "md": emit_md}[args.format](stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
