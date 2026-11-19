# Reading an Engram table kept in host memory

An Engram table can be kept in host (CPU-side) memory instead of GPU memory, because the rows a step reads are known from the token ids alone. This experiment compares, on one GPU, two ways of reading those rows:

* **way A**: the GPU reads each row from host memory itself (zero-copy). The table is in pinned host memory mapped into the GPU's address space, and a CUDA kernel gathers the rows over PCIe.
* **way B**: the CPU gathers the rows into a pinned staging buffer, then one copy sends them to the GPU.

Both are checked against the table in GPU memory on the trained model of [2026-10-21-engram-nanogpt](../2026-10-21-engram-nanogpt) (Hugging Face [littlemex/engram-nanogpt-wikitext103](https://huggingface.co/littlemex/engram-nanogpt-wikitext103), seed 1337): the rows read and the logits are bit-identical, and reading the neighbouring rows instead changes the logits.

## Results

Measured on one g7.2xlarge (NVIDIA RTX PRO 4500 Blackwell, PCIe Gen 5 x16, Intel Xeon 6975P-C with 8 vCPUs and a 480 MiB L3, 32 GiB of memory, Linux 7.0.0-1014-aws, driver 595.91.07, PyTorch 2.14.1+cu130, CUDA 13.2). One contiguous 256 MiB copy from pinned host memory to the GPU runs at 56.5 GB/s.

Reading 16,384 positions x 16 rows of 256 B (64 MiB), new random rows every call, from tables spread over all of their rows (`results/host-tables-*.json`, `fresh`):

| Table in host memory | Way A (GB/s) | Way B (GB/s) |
|---|---|---|
| 256 MiB | 50.7 | 24.0 |
| 512 MiB | 50.8 | 23.2 |
| 1 GiB | 25.9 | 22.0 |
| 2 GiB | 18.3 | 21.5 |
| 4 GiB | 15.3 | 21.2 |
| 8 GiB | 12.3 | 21.2 |
| 16 GiB | 10.0 | 20.7 |

The same read from a table in GPU memory runs at 303 GB/s at 16 GiB. For 16 positions (256 rows) from the 16 GiB table, way A takes 19.2 us and way B 36.6 us, with the row ids already on the CPU.

Way A slows down with the address range the rows are spread over, not with the bytes read (`stride`): the same 262,144 rows placed 256 B to 2 KiB apart (a 64 MiB to 512 MiB range) read at 50.5 GB/s, 4 KiB apart (1 GiB) at 25.9 GB/s and 16 KiB apart (4 GiB) at 15.3 GB/s. None of the following changes way A at any table size: 4 KiB or 2 MiB pages behind the host memory (transparent huge pages, hugetlbfs), the allocation (cudaHostAlloc, mmap with cudaHostRegister, cuMemCreate on the host NUMA node with a 2 MiB granularity), and the kernel's IOMMU default domain (translated, or `iommu=pt` in `results/host-tables-iommu-pt.json`).

`results/model-table-bf16*.json` hold the checks on the trained model and a prefill forward (8 x 512 tokens, bfloat16 autocast) with each way of reading.

## Files

| File | What it does |
|---|---|
| `engram_host_table.py` | The gather kernel and the allocations (compiled with `torch.utils.cpp_extension.load_inline`); the bit-for-bit checks on the trained model; read times for 1 to 65,536 positions; the prefill forward |
| `engram_host_table_pages.py` | Read speed against table size for every allocation (`sizes`, `fresh`), and the tests that separate bytes from address range (`pattern`, `stride`) |
| `results/` | The JSON each script wrote for the numbers above |

## Running it

On a Linux machine with an NVIDIA GPU, its driver, a CUDA toolkit with `nvcc` under `/usr/local/cuda`, Python 3.10 or later and at least 24 GiB of free memory. The extension is built by PyTorch with `ninja` and `nvcc`, both found through `PATH`:

```bash
git clone https://github.com/littlemex/zenmai-lab.git
cd zenmai-lab/experiments/2026-11-19-engram-host-table
python3 -m venv .venv
.venv/bin/pip install -q -r ../2026-10-21-engram-nanogpt/requirements.txt ninja
.venv/bin/hf download littlemex/engram-nanogpt-wikitext103 --revision e9758158ad0d180336bc059e20fecf29f67f445b --include "engram-seed1337/model.safetensors" --local-dir hub
export PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH
python engram_host_table.py --zenmai-dir ../2026-10-21-engram-nanogpt --checkpoint hub/engram-seed1337/model.safetensors --out out/model-table-bf16.json --bf16 --big-rows 0
sudo sysctl -q vm.nr_hugepages=4200
python engram_host_table_pages.py --out out/host-tables-256m-to-8g.json --sizes-mib 256,512,1024,2048,4096,8192 --tests sizes,pattern,stride,fresh --repeats 20
sudo sysctl -q vm.nr_hugepages=0
python engram_host_table_pages.py --out out/host-tables-16g.json --sizes-mib 16384 --tests sizes,fresh --modes gpu,cudaHostAlloc,register_4k,register_thp,cumem_host_numa,cpu_gather --repeats 20
```

`engram_host_table.py` ends with `RESULT: OK` when every way of reading matches the table in GPU memory and the shifted-rows control does not. The hugetlbfs pages reserved for the 256 MiB to 8 GiB run are released before the 16 GiB run, which leaves `register_hugetlb` out because 16 GiB of hugetlbfs pages and a 16 GiB table in ordinary pages do not both fit in 32 GiB.
