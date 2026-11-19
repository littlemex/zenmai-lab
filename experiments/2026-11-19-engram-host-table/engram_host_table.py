"""Where to keep an Engram table, and how to read its rows, on one GPU: in GPU memory, or in host memory read
either by the GPU directly (zero-copy over PCIe) or by the CPU (gather, then copy to the GPU).

Measures, on the trained Engram model of zenmai-lab (littlemex/engram-nanogpt-wikitext103):

1. correctness: every way of reading gives bit-identical rows, and bit-identical logits for a batch of windows;
2. the time to read the rows of N positions (16 rows each) with each method, for N from 1 to 65,536;
3. a prefill forward (batch x 512 tokens) with the rows read on the critical path, and read ahead on a second
   stream while the embedding and block 0 run, against the table in GPU memory.

Prints one JSON document; writes it to --out as well.
"""

import argparse
import contextlib
import copy
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void gather_rows(const float4* __restrict__ table, const int64_t* __restrict__ rows,
                            float4* __restrict__ out, int64_t n, int vec_per_row) {
  const int64_t total = n * vec_per_row;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = i / vec_per_row;
    const int c = (int)(i % vec_per_row);
    out[i] = table[rows[r] * vec_per_row + c];
  }
}

static void free_host(void* p) { cudaFreeHost(p); }

#include <sys/mman.h>
#include <unordered_map>
#include <mutex>
static std::unordered_map<void*, std::pair<void*, size_t>> g_maps;
static std::mutex g_mu;
static void free_registered(void* p) {
  std::lock_guard<std::mutex> l(g_mu);
  auto it = g_maps.find(p);
  cudaHostUnregister(p);
  if (it != g_maps.end()) { munmap(it->second.first, it->second.second); g_maps.erase(it); }
}

// The same table in host memory, but mmap'd by us and registered with cudaHostRegister, so that the page size
// behind it is chosen: mode 0 = 4 KiB pages (MADV_NOHUGEPAGE), 1 = transparent 2 MiB pages (MADV_HUGEPAGE),
// 2 = hugetlbfs 2 MiB pages (MAP_HUGETLB, needs vm.nr_hugepages).
#include <cuda.h>
// Host memory through the CUDA virtual memory API, as DeepEP allocates its Engram table: cuMemCreate on the
// host NUMA node, mapped at the allocation granularity, readable and writable by the GPU. Returns the GPU
// virtual address; the CPU does not touch it (fill and check it from the GPU).
static std::unordered_map<int64_t, std::pair<CUmemGenericAllocationHandle, size_t>> g_vmm;
std::vector<int64_t> alloc_cumem_host(int64_t bytes_wanted, int64_t numa_id) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = (int)numa_id;
  size_t gran = 0;
  TORCH_CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS, "granularity");
  const size_t bytes = ((size_t)bytes_wanted + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h;
  TORCH_CHECK(cuMemCreate(&h, bytes, &prop, 0) == CUDA_SUCCESS, "cuMemCreate HOST_NUMA failed");
  CUdeviceptr va = 0;
  TORCH_CHECK(cuMemAddressReserve(&va, bytes, gran, 0, 0) == CUDA_SUCCESS, "reserve");
  TORCH_CHECK(cuMemMap(va, bytes, 0, h, 0) == CUDA_SUCCESS, "map");
  CUmemAccessDesc acc = {};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = 0;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  TORCH_CHECK(cuMemSetAccess(va, bytes, &acc, 1) == CUDA_SUCCESS, "set access");
  g_vmm[(int64_t)va] = {h, bytes};
  return {(int64_t)va, (int64_t)bytes, (int64_t)gran};
}
void free_cumem_host(int64_t va) {
  auto it = g_vmm.find(va);
  TORCH_CHECK(it != g_vmm.end(), "unknown va");
  cudaDeviceSynchronize();
  cuMemUnmap((CUdeviceptr)va, it->second.second);
  cuMemAddressFree((CUdeviceptr)va, it->second.second);
  cuMemRelease(it->second.first);
  g_vmm.erase(it);
}
void copy_d2d(int64_t dst, int64_t src, int64_t bytes) {
  TORCH_CHECK(cudaMemcpy((void*)dst, (void*)src, (size_t)bytes, cudaMemcpyDefault) == cudaSuccess, "copy");
}

torch::Tensor alloc_registered(int64_t n_rows, int64_t dim, int64_t mode) {
  const size_t huge = 2ull << 20;
  const size_t bytes = (((size_t)n_rows * dim * sizeof(float)) + huge - 1) / huge * huge;
  const size_t span = mode == 2 ? bytes : bytes + huge;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | (mode == 2 ? MAP_HUGETLB : 0);
  void* base = mmap(nullptr, span, PROT_READ | PROT_WRITE, flags, -1, 0);
  TORCH_CHECK(base != MAP_FAILED, "mmap failed (mode ", mode, ")");
  void* p = mode == 2 ? base : (void*)(((uintptr_t)base + huge - 1) / huge * huge);
  if (mode == 0) madvise(p, bytes, MADV_NOHUGEPAGE);
  if (mode == 1) madvise(p, bytes, MADV_HUGEPAGE);
  memset(p, 0, bytes);
  TORCH_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterMapped) == cudaSuccess, "cudaHostRegister failed");
  { std::lock_guard<std::mutex> l(g_mu); g_maps[p] = {base, span}; }
  return torch::from_blob(p, {n_rows, dim}, free_registered, torch::TensorOptions().dtype(torch::kFloat32));
}

// A CPU tensor in pinned host memory that is mapped into the GPU's address space.
torch::Tensor alloc_mapped(int64_t n_rows, int64_t dim) {
  void* p = nullptr;
  const size_t bytes = (size_t)n_rows * dim * sizeof(float);
  TORCH_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocMapped) == cudaSuccess, "cudaHostAlloc failed for ", bytes, " bytes");
  return torch::from_blob(p, {n_rows, dim}, free_host, torch::TensorOptions().dtype(torch::kFloat32));
}

int64_t device_pointer(torch::Tensor host) {
  void* d = nullptr;
  TORCH_CHECK(cudaHostGetDevicePointer(&d, host.data_ptr(), 0) == cudaSuccess, "host memory is not mapped");
  return (int64_t)d;
}

// out[i, :] = table[rows[i], :]; table_ptr is a GPU address (device memory, or mapped host memory)
void gather(int64_t table_ptr, int64_t row_bytes, torch::Tensor rows, torch::Tensor out, int64_t max_blocks) {
  TORCH_CHECK(row_bytes % 16 == 0, "row bytes must be a multiple of 16");
  TORCH_CHECK(rows.is_cuda() && rows.dtype() == torch::kInt64 && rows.is_contiguous(), "rows: contiguous int64 on GPU");
  const int vpr = (int)(row_bytes / 16);
  const int64_t n = rows.numel();
  const int threads = 256;
  int64_t blocks = (n * vpr + threads - 1) / threads;
  if (max_blocks > 0 && blocks > max_blocks) blocks = max_blocks;
  if (blocks < 1) blocks = 1;
  gather_rows<<<(unsigned)blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const float4*)table_ptr, rows.data_ptr<int64_t>(), (float4*)out.data_ptr(), n, vpr);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gather launch failed");
}
"""
EXT_FUNCTIONS = ["alloc_mapped", "alloc_registered", "alloc_cumem_host", "free_cumem_host", "copy_d2d", "device_pointer", "gather"]
CPP_SRC = """
torch::Tensor alloc_mapped(int64_t n_rows, int64_t dim);
torch::Tensor alloc_registered(int64_t n_rows, int64_t dim, int64_t mode);
std::vector<int64_t> alloc_cumem_host(int64_t bytes_wanted, int64_t numa_id);
void free_cumem_host(int64_t va);
void copy_d2d(int64_t dst, int64_t src, int64_t bytes);
int64_t device_pointer(torch::Tensor host);
void gather(int64_t table_ptr, int64_t row_bytes, torch::Tensor rows, torch::Tensor out, int64_t max_blocks);
"""


class Reader:
    """Reads rows [N] of a table into a GPU tensor [N, dim], in one of four ways."""

    def __init__(self, ext, table_gpu: torch.Tensor, table_host: torch.Tensor, max_blocks: int):
        self.ext, self.gpu, self.host = ext, table_gpu, table_host
        self.dim = table_gpu.shape[1]
        self.row_bytes = self.dim * table_gpu.element_size()
        self.host_ptr = ext.device_pointer(table_host)
        self.max_blocks = max_blocks
        self.staging = None

    def out(self, n):
        return torch.empty((n, self.dim), dtype=torch.float32, device="cuda")

    def gpu_torch(self, rows):
        return self.gpu[rows]

    def gpu_kernel(self, rows):
        out = self.out(rows.numel())
        self.ext.gather(self.gpu.data_ptr(), self.row_bytes, rows, out, self.max_blocks)
        return out

    def host_zerocopy(self, rows):
        out = self.out(rows.numel())
        self.ext.gather(self.host_ptr, self.row_bytes, rows, out, self.max_blocks)
        return out

    def host_cpu_from_cpu_rows(self, rows_cpu):
        """The CPU gathers into a pinned staging buffer, then one copy to the GPU (on the current stream)."""
        n = rows_cpu.numel()
        if self.staging is None or self.staging.shape[0] < n:
            self.staging = torch.empty((n, self.dim), dtype=torch.float32, pin_memory=True)
        buf = self.staging[:n]
        torch.index_select(self.host, 0, rows_cpu, out=buf)
        out = self.out(n)
        out.copy_(buf, non_blocking=True)
        return out

    def host_cpu(self, rows):
        """Rows computed on the GPU: they come back to the CPU first, which waits for the GPU."""
        return self.host_cpu_from_cpu_rows(rows.cpu())


def timeit(fn, repeats: int, warmup: int = 3) -> float:
    """Median wall time of fn() in microseconds, the GPU drained before and after each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


class Given(nn.Module):
    """Stands in for Engram.table: returns rows already read, so Engram.forward runs unchanged."""

    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, rows):
        return self.value.view(*rows.shape, -1)


def forward(model, idx, read, prefetch: bool, side: torch.cuda.Stream, idx_cpu=None, hash_cpu=None):
    """The model's forward (one Engram layer, before block engram_layer), with the rows read by `read`.

    read(rows_gpu) -> rows on the GPU; with prefetch, the read runs on `side` while the embedding and the blocks
    before the Engram layer run on the current stream. read == "cpu" with idx_cpu and hash_cpu set is the CPU
    path that never waits for the GPU: the CPU hashes the token ids it already has, in a thread.
    """
    b, t = idx.shape
    engram_layer = model.config.engram_layer_ids[0]
    engram = model.engrams[str(engram_layer)]
    given = engram.table
    main = torch.cuda.current_stream()
    done = torch.cuda.Event()
    holder, thread = {}, None

    if hash_cpu is not None:
        def cpu_read():  # a new thread does not inherit inference mode
            with torch.inference_mode(), torch.cuda.stream(side):
                rows_cpu = hash_cpu(idx_cpu)[:, :, 0].reshape(-1).contiguous()
                holder["e"] = read.host_cpu_from_cpu_rows(rows_cpu)
                done.record(side)
        if prefetch:
            thread = threading.Thread(target=cpu_read)
            thread.start()
    x = model.wte(idx) + model.wpe(torch.arange(t, device=idx.device))
    if hash_cpu is None:
        rows = model.engram_hash(idx)[:, :, 0].reshape(-1).contiguous()
        if prefetch:
            ready = torch.cuda.Event()
            ready.record(main)
            side.wait_event(ready)
            with torch.cuda.stream(side):
                holder["e"] = read(rows)
                done.record(side)
    for layer_id, block in enumerate(model.blocks):
        if layer_id == engram_layer:
            if hash_cpu is not None and not prefetch:
                cpu_read()
            if prefetch or hash_cpu is not None:
                if thread is not None:
                    thread.join()
                main.wait_event(done)
                e = holder["e"]
                e.record_stream(main)
            else:
                e = read(rows)
            given.value = e
            x = engram(x, torch.empty((b, t, e.shape[0] // (b * t)), device="meta"))
        x = block(x)
    return model.lm_head(model.ln_f(x))


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenmai-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--big-rows", type=int, default=32 * 1024 * 1024, help="rows of the synthetic large table (0: skip)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=0, help="grid cap of the gather kernel (0: one thread per 16 B)")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--bf16", action="store_true", help="run the model under bfloat16 autocast, as in training")
    args = ap.parse_args()

    sys.path.insert(0, str(args.zenmai_dir))
    from engram_lm.evaluate import load

    torch.manual_seed(0)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.bf16 else contextlib.nullcontext

    def fwd(*a, **kw):
        with amp():
            return forward(*a, **kw)
    ext = load_inline("engram_host_gather", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                      functions=EXT_FUNCTIONS, extra_ldflags=["-lcuda"], verbose=False)
    model, _ = load(args.checkpoint, "cuda")
    model.eval()
    assert len(model.engrams) == 1, "this probe handles one Engram layer"
    engram = next(iter(model.engrams.values()))
    table_gpu = engram.table.weight.detach().contiguous()
    n_rows, dim = table_gpu.shape
    n_columns = engram.wkv.in_features // dim
    table_host = ext.alloc_mapped(n_rows, dim)
    table_host.copy_(table_gpu.cpu())
    engram.table = Given()
    props = torch.cuda.get_device_properties(0)
    report = {
        "gpu": props.name, "sm_count": props.multi_processor_count, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        "table": {"rows": n_rows, "dim": dim, "dtype": "float32", "bytes": n_rows * dim * 4, "rows_per_position": n_columns},
        "max_blocks": args.max_blocks, "bf16": args.bf16,
    }

    # 0. a reference: one large contiguous copy from pinned host memory to the GPU
    big = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
    dst = torch.empty_like(big, device="cuda")
    us = timeit(lambda: dst.copy_(big, non_blocking=True), 20)
    report["h2d_contiguous_GBps"] = big.numel() * 4 / us / 1e3
    del big, dst

    reader = Reader(ext, table_gpu, table_host, args.max_blocks)
    methods = {"gpu_torch": reader.gpu_torch, "gpu_kernel": reader.gpu_kernel,
               "host_zerocopy": reader.host_zerocopy, "host_cpu": reader.host_cpu}

    # 1. correctness: rows and logits
    ids = torch.randint(0, model.config.vocab_size, (args.batch, model.config.block_size), device="cuda")
    rows = model.engram_hash(ids)[:, :, 0].reshape(-1).contiguous()
    want = reader.gpu_torch(rows)
    report["rows_identical"] = {k: bool(torch.equal(f(rows), want)) for k, f in methods.items()}
    side = torch.cuda.Stream()
    hash_cpu = copy.deepcopy(model.engram_hash).cpu()
    ids_cpu = ids.cpu()
    report["cpu_hash_identical"] = bool(torch.equal(hash_cpu(ids_cpu)[:, :, 0].reshape(-1), rows.cpu()))
    ref = fwd(model, ids, reader.gpu_torch, False, side)
    cases = {
        "gpu_torch": dict(read=reader.gpu_torch, prefetch=False),
        "host_zerocopy": dict(read=reader.host_zerocopy, prefetch=False),
        "host_zerocopy_prefetch": dict(read=reader.host_zerocopy, prefetch=True),
        "host_cpu": dict(read=reader.host_cpu, prefetch=False),
        "host_cpu_hash_on_cpu": dict(read=reader, prefetch=False, idx_cpu=ids_cpu, hash_cpu=hash_cpu),
        "host_cpu_hash_on_cpu_prefetch": dict(read=reader, prefetch=True, idx_cpu=ids_cpu, hash_cpu=hash_cpu),
    }
    report["logits_identical"] = {k: bool(torch.equal(fwd(model, ids, side=side, **c), ref)) for k, c in cases.items()}
    # negative control: neighbouring rows must change the logits
    shifted = lambda r: reader.gpu_torch((r + 1) % n_rows)  # noqa: E731
    report["logits_identical"]["control_shift_rows_by_1"] = bool(torch.equal(fwd(model, ids, shifted, False, side), ref))

    # 2. the time to read the rows of N positions
    def read_times(table_rows: int, rd: Reader, names):
        out = []
        for n in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
            r = torch.randint(0, table_rows, (n * n_columns,), device="cuda")
            reps = args.repeats if n <= 4096 else max(5, args.repeats // 3)
            row = {"positions": n, "bytes": n * n_columns * dim * 4}
            r_cpu = r.cpu()
            for k in names:
                row[k + "_us"] = timeit(lambda f=getattr(rd, k): f(r), reps)
            if "host_cpu" in names:  # the CPU already holds the rows: nothing waits for the GPU first
                row["host_cpu_given_rows_us"] = timeit(lambda: rd.host_cpu_from_cpu_rows(r_cpu), reps)
            out.append(row)
        return out

    report["read_model_table"] = read_times(n_rows, reader, list(methods))
    if args.big_rows:
        big_gpu_rows = min(args.big_rows, n_rows)  # the GPU side is a stand-in; only host methods use the big table
        big_host = ext.alloc_mapped(args.big_rows, dim)
        big_host.normal_()
        big_reader = Reader(ext, table_gpu[:big_gpu_rows], big_host, args.max_blocks)
        report["big_table"] = {"rows": args.big_rows, "bytes": args.big_rows * dim * 4}
        report["read_big_table"] = read_times(args.big_rows, big_reader, ["host_zerocopy", "host_cpu"])
        del big_reader, big_host

    # 3. prefill: the forward pass, and the part before the Engram layer
    def before_engram():
        with amp():
            x = model.wte(ids) + model.wpe(torch.arange(ids.shape[1], device="cuda"))
            for layer_id, block in enumerate(model.blocks):
                if layer_id == model.config.engram_layer_ids[0]:
                    break
                x = block(x)
            return x
    report["prefill"] = {
        "batch": args.batch, "tokens": args.batch * ids.shape[1],
        "before_engram_us": timeit(before_engram, args.repeats),
        "read_alone_us": {k: timeit(lambda f=f: f(rows), args.repeats) for k, f in methods.items()},
        "forward_us": {k: timeit(lambda c=c: fwd(model, ids, side=side, **c), args.repeats) for k, c in cases.items()},
    }

    text = json.dumps(report, indent=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    ok = all(report["rows_identical"].values()) and report["cpu_hash_identical"] and all(
        v for k, v in report["logits_identical"].items() if not k.startswith("control")) and not report[
        "logits_identical"]["control_shift_rows_by_1"]
    print("RESULT:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
