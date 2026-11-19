"""Why reading a host-memory Engram table from the GPU (zero-copy) slows down as the table grows.

Four measurements with the gather kernel of engram_host_table.py, reading 16 random rows of 256 B per position
(16,384 positions, 64 MiB, unless said otherwise):

* sizes: tables of 256 MiB to 8 GiB in GPU memory and in host memory allocated five ways -- cudaHostAlloc;
  mmap + cudaHostRegister with 4 KiB pages, transparent 2 MiB pages, or hugetlbfs 2 MiB pages; and cuMemCreate on
  the host NUMA node (the CUDA virtual memory API, how DeepEP allocates its Engram table) -- each read by the GPU,
  and a 4 KiB-page table whose rows the CPU gathers and copies (cpu_gather);
* pattern: on one 8 GiB table, rows chosen so the bytes touched and the 4 KiB pages touched vary apart;
* stride: a fixed number of rows spaced 256 B to 16 KiB apart, to find what the cost follows;
* fresh: new random rows on every call, 16 to 16,384 positions, as every step of a model would read them.

Every host allocation is checked bit for bit against the GPU copy before it is timed. Prints one JSON document;
writes it to --out as well.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engram_host_table import CPP_SRC, CUDA_SRC, EXT_FUNCTIONS, timeit  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

DIM, COLS, ROW_BYTES = 64, 16, 256


def meminfo() -> dict:
    keep = ("AnonHugePages", "HugePages_Total", "HugePages_Free")
    return {k: v.strip() for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines()) if k in keep}


class HostTable:
    """A table of n_rows x 64 float32 in host memory, filled from `src` (a GPU tensor of the same shape)."""

    def __init__(self, ext, mode: str, src: torch.Tensor):
        self.ext, self.mode, self.keep, self.va, self.staging = ext, mode, None, None, None
        n_rows = src.shape[0]
        if mode == "cpu_gather":
            t = ext.alloc_registered(n_rows, DIM, 0)
            ext.copy_d2d(ext.device_pointer(t), src.data_ptr(), n_rows * ROW_BYTES)
            self.keep, self.ptr, self.granularity, self.staging = t, None, None, None
        elif mode == "cumem_host_numa":
            va, nbytes, gran = ext.alloc_cumem_host(n_rows * ROW_BYTES, 0)
            ext.copy_d2d(va, src.data_ptr(), n_rows * ROW_BYTES)
            self.va, self.ptr, self.granularity = va, va, gran
        else:
            t = ext.alloc_mapped(n_rows, DIM) if mode == "cudaHostAlloc" else ext.alloc_registered(
                n_rows, DIM, {"register_4k": 0, "register_thp": 1, "register_hugetlb": 2}[mode])
            ext.copy_d2d(ext.device_pointer(t), src.data_ptr(), n_rows * ROW_BYTES)
            self.keep, self.ptr, self.granularity = t, ext.device_pointer(t), None
        torch.cuda.synchronize()

    def read(self, rows: torch.Tensor) -> torch.Tensor:
        if self.mode == "cpu_gather":
            return self.read_cpu(rows.cpu())
        out = torch.empty((rows.numel(), DIM), dtype=torch.float32, device="cuda")
        self.ext.gather(self.ptr, ROW_BYTES, rows, out, 0)
        return out

    def read_cpu(self, rows_cpu: torch.Tensor) -> torch.Tensor:
        """The CPU gathers the rows (it already holds their ids) into pinned memory; one copy to the GPU."""
        n = rows_cpu.numel()
        if self.staging is None or self.staging.shape[0] < n:
            self.staging = torch.empty((n, DIM), dtype=torch.float32, pin_memory=True)
        torch.index_select(self.keep, 0, rows_cpu, out=self.staging[:n])
        out = torch.empty((n, DIM), dtype=torch.float32, device="cuda")
        out.copy_(self.staging[:n], non_blocking=True)
        return out

    def free(self):
        if self.va is not None:
            self.ext.free_cumem_host(self.va)
        self.keep = None


def gbps(n_rows_read: int, us: float) -> float:
    return n_rows_read * ROW_BYTES / us / 1e3


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes-mib", default="256,1024,2048,4096,8192")
    ap.add_argument("--modes", default="gpu,cudaHostAlloc,register_4k,register_thp,register_hugetlb,cumem_host_numa,cpu_gather")
    ap.add_argument("--tests", default="sizes,pattern,stride")
    ap.add_argument("--positions", type=int, default=16384)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    ext = load_inline("engram_host_gather", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=EXT_FUNCTIONS,
                      extra_ldflags=["-lcuda"], verbose=False)
    n_read = args.positions * COLS
    tests = args.tests.split(",")
    report = {"gpu": torch.cuda.get_device_name(0), "positions": args.positions, "rows_per_position": COLS,
              "row_bytes": ROW_BYTES, "read_bytes": n_read * ROW_BYTES, "sizes": [], "pattern": [], "stride": []}

    def measure(table_read, rows):
        return timeit(lambda: table_read(rows), args.repeats)

    if "sizes" in tests:
        for mib in map(int, args.sizes_mib.split(",")):
            n_rows = mib * 2**20 // ROW_BYTES
            dev = torch.empty((n_rows, DIM), dtype=torch.float32, device="cuda").normal_()
            rows = torch.randint(0, n_rows, (n_read,), device="cuda")
            want = dev[rows]
            us = measure(lambda r: dev[r], rows)
            report["sizes"].append({"table_mib": mib, "memory": "gpu", "us": us, "GBps": gbps(n_read, us)})
            print(f"{mib:>6} MiB {'gpu':>17}: {gbps(n_read, us):6.1f} GB/s", flush=True)
            for mode in args.modes.split(","):
                if mode == "gpu":  # measured above
                    continue
                table = HostTable(ext, mode, dev)
                assert torch.equal(table.read(rows), want), mode
                if mode == "cpu_gather":
                    rows_cpu = rows.cpu()
                    us = measure(table.read_cpu, rows_cpu)
                else:
                    us = measure(table.read, rows)
                run = {"table_mib": mib, "memory": mode, "us": us, "GBps": gbps(n_read, us),
                       "granularity": table.granularity, "meminfo": meminfo()}
                report["sizes"].append(run)
                print(f"{mib:>6} MiB {mode:>17}: {run['GBps']:6.1f} GB/s  {run['meminfo']}  gran {table.granularity}", flush=True)
                table.free()
                del table
                gc.collect()
            del dev, want
            torch.cuda.empty_cache()

    if "fresh" in tests:
        # every call reads a new set of random rows, as every step of a model would
        report["fresh"] = []
        for mib in map(int, args.sizes_mib.split(",")):
            n_rows = mib * 2**20 // ROW_BYTES
            dev = torch.empty((n_rows, DIM), dtype=torch.float32, device="cuda").normal_()
            for mode in args.modes.split(","):
                table = HostTable(ext, mode, dev) if mode != "gpu" else None
                for positions in (16, 256, 4096, 16384):
                    sets = [torch.randint(0, n_rows, (positions * COLS,), device="cuda") for _ in range(args.repeats + 3)]
                    if mode == "cpu_gather":
                        sets = [r.cpu() for r in sets]
                        fn = table.read_cpu
                    else:
                        fn = (lambda r: dev[r]) if mode == "gpu" else table.read
                    it = iter(sets)
                    us = timeit(lambda: fn(next(it)), args.repeats, warmup=3)
                    run = {"table_mib": mib, "memory": mode, "positions": positions, "us": us,
                           "GBps": gbps(positions * COLS, us)}
                    report["fresh"].append(run)
                    print(f"fresh {mib:>6} MiB {mode:>14} positions {positions:>6}: {us:9.1f} us {run['GBps']:6.1f} GB/s", flush=True)
                if table is not None:
                    table.free()
                del table
                gc.collect()
            del dev
            torch.cuda.empty_cache()

    if "pattern" in tests or "stride" in tests:
        n_rows = 8 * 2**30 // ROW_BYTES
        dev = torch.empty((n_rows, DIM), dtype=torch.float32, device="cuda").normal_()
        for mode in ("register_4k", "cumem_host_numa"):
            table = HostTable(ext, mode, dev)
            if "pattern" in tests:
                # dense: R rows from the first R rows; one_per_page: the first row of R different 4 KiB pages
                for r in (4096, 16384, 65536, 262144, 1048576, n_rows // 16):
                    pick = torch.randint(0, r, (n_read,), device="cuda")
                    for name, rows in (("dense", pick), ("one_per_page", pick * 16)):
                        us = measure(table.read, rows)
                        run = {"memory": mode, "pattern": name, "distinct_rows": r, "bytes_touched": r * ROW_BYTES,
                               "pages_4k_touched": r if name == "one_per_page" else max(1, r // 16),
                               "us": us, "GBps": gbps(n_read, us)}
                        report["pattern"].append(run)
                        print(f"{mode:>16} {name:>13} rows {r:>8} pages {run['pages_4k_touched']:>8}: {run['GBps']:6.1f} GB/s", flush=True)
            if "stride" in tests:
                # 262,144 distinct rows, spaced `stride` bytes apart (each row is 256 B)
                r = 262144
                pick = torch.randint(0, r, (n_read,), device="cuda")
                for stride in (256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 2**21):
                    if r * stride > n_rows * ROW_BYTES:
                        continue
                    rows = pick * (stride // ROW_BYTES)
                    us = measure(table.read, rows)
                    run = {"memory": mode, "distinct_rows": r, "stride_bytes": stride, "span_mib": r * stride / 2**20,
                           "us": us, "GBps": gbps(n_read, us)}
                    report["stride"].append(run)
                    print(f"{mode:>16} stride {stride:>8} B span {run['span_mib']:>8.0f} MiB: {run['GBps']:6.1f} GB/s", flush=True)
            table.free()
            del table
            gc.collect()
        del dev
        torch.cuda.empty_cache()

    text = json.dumps(report, indent=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
