# ParGraph: Parallel Graph Analytics from SIMD to MPI

A systems-engineering study of core graph analytics kernels (**BFS, PageRank, Connected Components, Triangle Counting**) implemented at four levels of parallelism (**SIMD, OpenMP, CUDA, MPI**) and benchmarked on graphs with very different structure.

> **Research question:** *Which parallelization strategy wins on which kernel and which graph structure, and why?*

**Status:** 🚧 Phase 1 — serial baselines. See the [Roadmap](#roadmap). Results tables are templates until you fill them with your own measurements.

**Implementation rule:** finish **serial** logic for **all four kernels** (and validate it) before writing any OpenMP, SIMD, CUDA, or MPI variant.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Kernels](#kernels)
3. [Roadmap](#roadmap)
4. [Parallelization Matrix](#parallelization-matrix)
5. [Repository Layout](#repository-layout)
6. [Requirements](#requirements)
7. [Build](#build)
8. [Datasets](#datasets)
9. [Usage](#usage)
10. [Validation](#validation)
11. [Benchmarking](#benchmarking)
12. [Profiling](#profiling)
13. [Results](#results)
14. [Design Notes](#design-notes)


---

## Motivation

Graphs model social networks, web links, road maps, fraud rings, citation networks, and biological interaction networks. Questions like *who is most influential, what is reachable, which nodes form a group, and how tightly clustered is a network* reduce to a small set of kernels. Real graphs have millions to billions of edges, so serial code becomes too slow for large inputs or repeated queries.

These kernels are also hard to parallelize, which is the point of the project:

- **Irregular memory access:** following edges means random reads, so performance is bound by memory latency and bandwidth, not FLOPs.
- **Power-law degree distribution:** a few hub vertices have huge neighbor lists, causing load imbalance.
- **Structure changes the best strategy:** social graphs have few, huge BFS levels; road networks have thousands of tiny ones.
- **Different kernels stress different things:** frontier management (BFS), iterative memory-bound sweeps (PageRank), race-prone label updates (CC), and set intersection (triangle counting).

## Kernels

| Kernel | Question it answers | Serial baseline | Parallel character |
|---|---|---|---|
| **BFS** | How many hops from a source to every vertex? | Queue-based, O(V+E) | Level-synchronous, irregular, frontier-bound |
| **PageRank** | Which vertices are most important? | Power iteration (pull), O(E) per iteration | Regular sweep, memory-bandwidth-bound |
| **Connected Components** | Which vertices are reachable from each other? | Union-find / label propagation | Race-prone updates, convergence behavior |
| **Triangle Counting** | How tightly clustered is the graph? | Sorted-list intersection over ordered edges | Compute-leaning, skew-sensitive, SIMD-friendly |

*Stretch kernels:* SSSP (delta-stepping) and Betweenness Centrality.

## Roadmap

Work proceeds in order. Do not start a later phase until the earlier one is correct on the tiny graphs and at least one mid-size graph.

| Phase | Focus | Done when |
|---|---|---|
| **0 — Infra** | CSR load/build, generators, CLI, timers, tiny-graph tests | Graphs load; `--variant serial` runs end-to-end |
| **1 — Serial (all kernels)** | BFS, PageRank, CC, Triangle Counting — **serial only** | Each kernel matches NetworkX / hand checks; `--validate` passes |
| **2 — OpenMP** | Shared-memory parallel for all four kernels | Speedup vs. serial; no races; validate vs. serial |
| **3 — SIMD** | Vectorized paths (priority: TC intersection, bitmaps) | Measurable win or documented non-win vs. scalar |
| **4 — CUDA** | GPU variants on the RTX 5060 | Correct vs. serial; H2D separated from kernel time |
| **5 — MPI** | Distributed / hybrid MPI+OpenMP | Correct partitions; compute vs. communication reported |
| **6 — Measure & write** | Full benchmark + profile suite, fill Results, writeup | Tables and figures in `results/`; analysis in `docs/writeup.md` |

Phase 1 is intentionally **breadth-first**: one correct serial implementation of every algorithm before any parallel code. Parallel variants share the same interfaces and use the serial results as the correctness oracle.

## Parallelization Matrix

| Kernel | Serial | OpenMP | SIMD | CUDA | MPI |
|---|---|---|---|---|---|
| BFS | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
| PageRank | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
| Connected Components | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
| Triangle Counting | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |

Cells move left → right only after Phase 1 is complete for that kernel (and preferably for all kernels).

## Repository Layout

```
pargraph/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── .gitignore
├── include/
│   ├── graph.hpp            # CSR structure, loader, validators
│   ├── bitmap.hpp           # bitmap frontier + SIMD helpers
│   ├── intersect.hpp        # sorted-set intersection (scalar + SIMD)
│   ├── timer.hpp
│   └── kernels.hpp          # common kernel interfaces
├── src/
│   ├── graph/
│   │   ├── csr_builder.cpp
│   │   ├── generators.cpp   # kronecker, uniform, grid
│   │   └── reorder.cpp      # degree sort, RCM, random
│   ├── bfs/        {serial.cpp, omp.cpp, simd.cpp, cuda.cu, mpi.cpp}
│   ├── pagerank/   {serial.cpp, omp.cpp, simd.cpp, cuda.cu, mpi.cpp}
│   ├── cc/         {serial.cpp, omp.cpp, simd.cpp, cuda.cu, mpi.cpp}
│   ├── tc/         {serial.cpp, omp.cpp, simd.cpp, cuda.cu, mpi.cpp}
│   └── main.cpp             # unified CLI driver
├── tools/
│   ├── convert_graph.cpp    # text / .mtx / .gr -> binary CSR
│   └── generate.cpp         # synthetic graph generator
├── tests/
│   ├── tiny_graphs/         # hand-checkable graphs with known answers
│   └── test_*.cpp           # per-kernel unit tests
├── scripts/
│   ├── download_datasets.sh
│   ├── crosscheck_networkx.py
│   ├── run_benchmarks.sh
│   ├── run_scaling.sh
│   └── plot_results.py      # regenerates every figure
├── data/                    # downloaded + converted graphs (git-ignored)
├── results/                 # raw CSVs + figures (committed); profiler dumps ignored
├── docker/Dockerfile
└── docs/writeup.md          # design decisions, profiling analysis
```

## Requirements

**Target machine (this project)**

| Component | Spec |
|---|---|
| OS | Windows 10/11 |
| CPU | 13th Gen Intel Core i7, 14 cores |
| Caches | L1 1.2 MB · L2 11.5 MB · L3 24 MB |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 5060, 8 GB |

With 16 GB host RAM and 8 GB VRAM, prefer LiveJournal / Kronecker ≤22 for routine runs; USA-road and Kronecker ≥24 need careful memory budgeting (CSR neighbors ≈ `edges × 4` bytes with 32-bit IDs).

**Required**

- CMake 3.20+
- C++17 compiler: MSVC (Visual Studio 2022) or Clang-cl; MinGW-w64 also works for CPU-only builds
- Git Bash or WSL2 for the `scripts/*.sh` helpers (or run the equivalent commands by hand)

**Optional (per variant)**

| Feature | Requirement |
|---|---|
| SIMD | CPU with AVX2 (AVX-512 optional). Check in PowerShell: `(Get-CimInstance Win32_Processor).Caption` and confirm AVX2 in CPU-Z / `coreinfo -f` |
| CUDA | NVIDIA GPU + CUDA Toolkit 11.8+ (Nsight tools install with the toolkit / NVIDIA drivers) |
| MPI | Microsoft MPI (MS-MPI) or Intel MPI on Windows (`mpiexec -help`) |

**Analysis and plotting**

- Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `networkx`
- Profilers: see [Profiling](#profiling)

## Build

Out-of-source CMake build on Windows (PowerShell). Phase 1 only needs the serial CPU target; enable CUDA/MPI later when you reach those roadmap phases.

```powershell
# Configure (serial + OpenMP-ready; adjust generator if needed)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DPARGRAPH_ENABLE_CUDA=OFF `
  -DPARGRAPH_ENABLE_MPI=OFF

# Build Release
cmake --build build --config Release

# Run (exe under build/Release with the VS generator)
.\build\Release\pargraph.exe --help
```

**CMake options**

| Option | Default | Meaning |
|---|---|---|
| `PARGRAPH_ENABLE_CUDA` | `OFF` | Build CUDA kernels (needs CUDA Toolkit + RTX 5060 drivers) |
| `PARGRAPH_ENABLE_MPI` | `OFF` | Build MPI variants (needs MS-MPI or Intel MPI) |
| `PARGRAPH_ENABLE_SIMD` | `ON` | Compile AVX2 code paths (safe to leave on; unused until Phase 3) |
| `PARGRAPH_BUILD_TESTS` | `ON` | Build `tests/test_*.cpp` targets |
| `PARGRAPH_BUILD_TOOLS` | `ON` | Build `convert_graph` and `generate` |

**Ninja + Clang-cl** (optional, faster incremental builds):

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=clang-cl `
  -DPARGRAPH_ENABLE_CUDA=OFF -DPARGRAPH_ENABLE_MPI=OFF
cmake --build build
.\build\pargraph.exe --help
```

**Sanitizers** (Clang/MSVC AddressSanitizer; use before trusting parallel results in later phases):

```powershell
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_CXX_FLAGS="/fsanitize=address"
cmake --build build-asan
```

**Tests**

```powershell
cd build
ctest -C Release --output-on-failure
```

Build trees (`build/`, `build-*/`) and downloaded graphs under `data/` are git-ignored — see [`.gitignore`](.gitignore).

## Datasets

Use at least one graph from each structural class. Download into `data/raw/`, then convert to binary CSR (see below).

| Graph | Class | Approx. size | Source | Download |
|---|---|---|---|---|
| LiveJournal (`soc-LiveJournal1`) | Real social, power-law | ~4.8M V, ~69M E | [SNAP](https://snap.stanford.edu/data/soc-LiveJournal1.html) | [soc-LiveJournal1.txt.gz](https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz) (~1 GB compressed) |
| Orkut (`com-Orkut`) | Real social, power-law | ~3.1M V, ~117M E | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) | [com-orkut.ungraph.txt.gz](https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz) |
| USA road (DIMACS) | High diameter, low degree | up to ~24M V | [DIMACS 9th Challenge](https://www.diag.uniroma1.it/challenge9/download.shtml) | see road table below |
| Kronecker (scale 20–26) | Synthetic power-law | 2^scale V | generated | `tools/generate` (no download) |
| 2D grid | Synthetic road-like | any | generated | `tools/generate` (no download) |

**DIMACS USA road networks** (distance `.gr.gz`; topology is what matters for BFS/CC/TC). Prefer smaller regions first on 16 GB RAM — full USA is tight.

| Region | Nodes | Arcs | Download |
|---|---|---|---|
| NY (starter) | ~0.26M | ~0.73M | [USA-road-d.NY.gr.gz](https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz) |
| BAY | ~0.32M | ~0.80M | [USA-road-d.BAY.gr.gz](https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.BAY.gr.gz) |
| COL | ~0.44M | ~1.1M | [USA-road-d.COL.gr.gz](https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.COL.gr.gz) |
| FLA | ~1.1M | ~2.7M | [USA-road-d.FLA.gr.gz](https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.FLA.gr.gz) |
| CAL | ~1.9M | ~4.7M | [USA-road-d.CAL.gr.gz](https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.CAL.gr.gz) |
| USA (full) | ~23.9M | ~58.3M | [USA-road-d.USA.gr.gz](https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz) (~335 MB) |

Index of all regions: [challenge9/download.shtml](https://www.diag.uniroma1.it/challenge9/download.shtml). Pattern: `…/USA-road-d/USA-road-d.<NAME>.gr.gz`.

**Download (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null

# LiveJournal (~1 GB gz → ~1.1 GB text)
Invoke-WebRequest -Uri "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz" `
  -OutFile "data\raw\soc-LiveJournal1.txt.gz"

# Orkut (large; optional until TC/CC benches)
Invoke-WebRequest -Uri "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz" `
  -OutFile "data\raw\com-orkut.ungraph.txt.gz"

# Small road starter (NY)
Invoke-WebRequest -Uri "https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz" `
  -OutFile "data\raw\USA-road-d.NY.gr.gz"

# Decompress (.NET / tar on Win10+)
tar -xf data\raw\soc-LiveJournal1.txt.gz -C data\raw
tar -xf data\raw\USA-road-d.NY.gr.gz -C data\raw
```

Suggested order on this machine: **NY/BAY road → LiveJournal → Kronecker-20/22 (generated) → Orkut / larger roads** once serial + convert tools work.

**Convert to the binary CSR format** (done once; loads in seconds afterward):

```powershell
# After tools are built (see Build)
.\build\Release\convert_graph.exe --in data\raw\soc-LiveJournal1.txt --format edgelist `
  --out data\livejournal.csr
.\build\Release\convert_graph.exe --in data\raw\USA-road-d.NY.gr --format gr `
  --out data\usa_ny.csr
```

Formats: `edgelist` (whitespace pairs, `#` comments), `mtx` (Matrix Market / SuiteSparse), `gr` (DIMACS).

**Per-kernel preprocessing**

| Kernel | Graph treatment |
|---|---|
| BFS, CC | Undirected (reverse edges added), deduplicated, no self-loops |
| PageRank | Directed (keep original direction); build both in-edge and out-edge CSR |
| Triangle Counting | Undirected, sorted adjacency, and orientation by degree (or ID) so each triangle is counted once. Optional degree-based reordering |

Check dataset sizes against your RAM and GPU memory. The CSR neighbor array takes roughly `edges × 4 bytes` with 32-bit IDs. LiveJournal neighbors alone are ~280 MB; Orkut ~470 MB; full USA road is larger once made undirected.

## Usage

All kernels and variants share one CLI:

```bash
./build/pargraph --kernel <bfs|pagerank|cc|tc> \
                 --variant <serial|omp|simd|cuda|mpi> \
                 --graph data/livejournal.csr \
                 --threads 16 --trials 5 --validate \
                 --output results/lj_bfs_omp.csv
```

**Common flags**

| Flag | Meaning |
|---|---|
| `--kernel` | `bfs`, `pagerank`, `cc`, `tc` |
| `--variant` | Implementation level (and sub-variant, e.g. `omp_diropt`) |
| `--threads <n>` | OpenMP threads (default: all cores) |
| `--trials <n>` | Repetitions after warm-up |
| `--validate` | Check output against serial/reference |
| `--reorder <none\|degree\|rcm\|random>` | Vertex reordering before the kernel |
| `--output <csv>` | Raw per-run results |

**Kernel-specific flags**

| Kernel | Flags |
|---|---|
| BFS | `--sources 64` (random roots, fixed `--seed`), `--alpha 15 --beta 18` (direction switching), `--per-level` |
| PageRank | `--damping 0.85 --tolerance 1e-4 --max-iters 100` |
| CC | `--algo <labelprop\|sv\|afforest>` |
| TC | `--intersect <merge\|galloping\|simd\|bitmap>`, `--orient <degree\|id>` |

**Examples**

```bash
# Direction-optimizing BFS
./build/pargraph --kernel bfs --variant omp_diropt --graph data/kron22.csr --sources 64 --validate

# PageRank on GPU
./build/pargraph --kernel pagerank --variant cuda --graph data/livejournal.csr --validate

# SIMD triangle counting with degree orientation
./build/pargraph --kernel tc --variant simd --intersect simd --orient degree \
                 --graph data/orkut.csr --validate

# Distributed connected components (hybrid MPI + OpenMP)
mpirun -np 4 --map-by node:PE=8 \
    ./build/pargraph --kernel cc --variant mpi --threads 8 --graph data/kron24.csr
```


## Validation

Correctness comes before speed. Every parallel variant is checked against the serial baseline, and the serial baseline against an independent reference.

| Kernel | Check |
|---|---|
| **BFS** | Exact distance match vs. serial. Graph500-style tree check (each parent one level closer; every edge spans at most one level). Reached sets identical |
| **PageRank** | L1/L∞ difference vs. serial under the same tolerance, same iteration count. Scores sum to ~1. Parallel floating-point order can cause tiny differences, so compare within a tolerance |
| **CC** | Same partition as serial (compare component sizes and verify labels agree across every edge). Component labels may differ, so compare partitions rather than label values |
| **TC** | Exact total match (integer result). Cross-check small graphs with NetworkX; compare large-graph totals with SNAP's published statistics where available |

General checks:

1. Tiny hand-verified graphs in `tests/tiny_graphs/` with known answers for every kernel.
2. NetworkX cross-check on a ~100K-edge subgraph: `python scripts/crosscheck_networkx.py --graph data/small.csr`
3. CSR sanity: offsets non-decreasing, last offset equals edge count, neighbor IDs in range, sorted adjacency, and symmetric edges for undirected graphs.
4. Run the tests under sanitizers (`-fsanitize=address,undefined`; `thread` sanitizer for OpenMP races) before benchmarking.

## Benchmarking

**Methodology**

- Warm-up run first, then the median (with min/max) over multiple trials.
- **BFS:** 64 random roots from the largest connected component; report harmonic-mean TEPS (Graph500/GAP convention).
- **PageRank:** time per iteration and total time to convergence at a fixed tolerance.
- **CC:** total time to convergence; report iteration or round count.
- **TC:** total time and intersection rate (edge checks per second).
- Preprocessing (parsing, CSR build, reordering, orientation, host-to-device copies, MPI distribution) is reported **separately** from kernel time.
- Record machine specs, compiler flags, git commit, and thread/rank counts with every result.

**Suggested thread / rank defaults on this machine**

| Variant | Suggestion |
|---|---|
| OpenMP | `--threads 14` (match physical cores); also sweep 1, 2, 4, 7, 14 for strong scaling |
| CUDA | single GPU (RTX 5060, 8 GB); keep working set under ~6–7 GB to leave headroom |
| MPI hybrid | `mpiexec -n 2` with `--threads 7` per rank (or 1 rank × 14 threads) for local multi-process experiments |

## Profiling

Profiling answers *why* a variant wins or loses on a given graph — not just how fast it is. Capture CPU, memory, and GPU counters alongside every benchmark CSV you keep.

**What to look for (by kernel)**

| Kernel | Likely bottleneck | Counters / views to capture |
|---|---|---|
| BFS | Frontier irregularity, atomics, level launch overhead | Cache misses, atomic contention, time per level, GPU kernel-launch count on road graphs |
| PageRank | DRAM bandwidth | Achieved GB/s vs. peak, LLC miss rate, stalled cycles on loads |
| CC | Atomic CAS / `atomicMin`, convergence rounds | Contended atomics, rounds to fixpoint, load imbalance |
| TC | Set-intersection throughput, degree skew | SIMD instruction mix, time per edge bucketed by degree, branch mispredicts |

**Windows tooling**

| Layer | Tool | Use for |
|---|---|---|
| CPU / OpenMP / SIMD | [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html) | Hotspots, memory-access, threading, microarchitecture (L1/L2/L3) on the 13th-gen i7 |
| CPU (lightweight) | Visual Studio Diagnostic Tools / Concurrency Visualizer | Quick wall-clock and OpenMP oversubscription checks |
| CUDA | Nsight Systems (`nsys`) | Timeline: H2D copies vs. kernel time vs. idle |
| CUDA | Nsight Compute (`ncu`) | Occupancy, warp stall reasons, achieved vs. peak bandwidth/FLOPs |
| MPI | Intel Trace Analyzer / MS-MPI + ETW, or rank-local timers | Compute vs. wait / communication share |

Linux `perf` / `likwid` are not native on Windows; use VTune (or profile under WSL2 if you need `perf` specifically).

**Example commands**

```powershell
# Nsight Systems — full GPU run timeline
nsys profile -o results/pr_cuda_lj --stats true `
  .\build\Release\pargraph.exe --kernel pagerank --variant cuda `
  --graph data\livejournal.csr --trials 5

# Nsight Compute — kernel-level GPU metrics (PageRank pull kernel)
ncu --set full -o results\pr_cuda_lj_ncu `
  .\build\Release\pargraph.exe --kernel pagerank --variant cuda `
  --graph data\livejournal.csr --trials 1

# VTune hotspot + memory access (OpenMP BFS)
vtune -collect hotspots -result-dir results\bfs_omp_hs -- `
  .\build\Release\pargraph.exe --kernel bfs --variant omp `
  --graph data\livejournal.csr --threads 14 --sources 64

vtune -collect memory-access -result-dir results\bfs_omp_mem -- `
  .\build\Release\pargraph.exe --kernel bfs --variant omp `
  --graph data\livejournal.csr --threads 14 --sources 64
```

**How to report profiles**

1. Attach profiler output (or a short summary) next to the matching `results/*.csv` row.
2. For each kernel × graph, state the primary bottleneck (compute / bandwidth / latency / sync / launch overhead).
3. Tie findings back to the research question: e.g. why CUDA loses on USA-road BFS (many tiny levels) but wins on LiveJournal PageRank (bulk memory traffic).
4. Longer write-ups and annotated screenshots live in [`docs/writeup.md`](docs/writeup.md).

## Results

> Fill in with your own measurements. State machine specs for every table.

**Test machine:** 13th Gen Intel Core i7 (14 cores) · L1 1.2 MB / L2 11.5 MB / L3 24 MB · 16 GB RAM · NVIDIA RTX 5060 8 GB · Windows

### Serial baselines

Raw per-trial rows + metadata: [`results/lj_bfs_serial.csv`](results/lj_bfs_serial.csv), [`results/lj_bfs_serial.csv.meta.json`](results/lj_bfs_serial.csv.meta.json). Kernel time excludes CSR load.

| Kernel | Graph | Variant | Setup | Median kernel (ms) | Harmonic-mean TEPS | Notes |
|---|---|---|---|---|---|---|
| BFS | LiveJournal (`data/livejournal.csr`) | serial | 8 random roots, seed=1, 1 warmup + 3 trials/root, `--validate` | 1241.73 | 5.19e7 | Directed CSR (V=4,847,571, E=68,475,391). Load ~152 ms. 6/8 roots reach ~4.40M verts / ~67.7M edges examined (~1.2–2.0 s); 2/8 roots have out-degree 0 (`visited=1`, ~24 ms init only). Commit `08a5912`. |
| BFS | LiveJournal (`data/livejournal.csr`) | omp (no-atomic discovery + reduction-tree merge) | 14 threads, 8 random roots, seed=1, 1 warmup + 3 trials/root, `--validate` (Graph500 + exact dist vs serial) | 1527.88 | 4.39e7 | Same graph/roots as serial. Load ~126 ms. ≈ **0.81×** vs serial median (candidate sort/dedupe dominates on this power-law graph; post-fix rerun). Validated exact dist vs serial. See [`results/lj_bfs_omp.csv`](results/lj_bfs_omp.csv) + [`.meta.json`](results/lj_bfs_omp.csv.meta.json). |
| BFS | LiveJournal (`data/livejournal.csr`) | omp_atomic (atomic visited TAS + thread-local next) | 14 threads, 8 random roots, seed=1, 1 warmup + 3 trials/root, `--validate` (Graph500 + exact dist vs serial) | 173.99 | 3.84e8 | Same graph/roots as serial. Load ~264 ms. Speedup ≈ **7.14×** vs serial median. Plain load before CAS reduces atomic traffic on already-visited neighbors. See [`results/lj_bfs_omp_atomic.csv`](results/lj_bfs_omp_atomic.csv) + [`.meta.json`](results/lj_bfs_omp_atomic.csv.meta.json). |
| BFS | LiveJournal (`data/livejournal.csr`) | mpi_atomic (hybrid MPI + OpenMP, 1D vertex ownership, atomic TAS) | `mpiexec -n 2`, 7 threads/rank, 8 random roots, seed=1, 1 warmup + 3 trials/root, `--validate` (Graph500 + exact dist/edges vs serial) | 246.66 | 2.65e8 | Same graph/roots as serial. Load ~160 ms. 1D edge slice ~51 ms, excluded from kernel time. Speedup ≈ **5.03×** vs serial median. Slowest rank: median compute 192 ms, comm 38 ms (**16.7%** of that split, including MPI wait). Discovery payload ~22.8 MB/trial on the large roots. See [`results/lj_bfs_mpi_atomic.csv`](results/lj_bfs_mpi_atomic.csv) + [`.meta.json`](results/lj_bfs_mpi_atomic.csv.meta.json). |
| CC | LiveJournal (`data/livejournal.csr`) | serial union-find | 1 warmup + 3 trials, `--validate` | 10457 | — | WCC via union over directed arcs (V=4,847,571, E=68,475,391). Load ~208 ms. **1876** components, largest **4,843,953**. Debug build, commit `217f330`. See [`results/lj_cc_serial.csv`](results/lj_cc_serial.csv) + [`.meta.json`](results/lj_cc_serial.csv.meta.json). |
| TC | Synth-5k (`tests/tiny_graphs/tc_synth_5k.csr`) | serial forward (degree orient, merge) | 1 warmup + 5 trials, `--validate` | 2.691 | — | Undirected (V=5000, E=89658), **6924** triangles. Orient ~0.91 ms (excluded). Median intersect rate ≈ 1.67e7 ops/s. See [`results/tc_synth5k_serial.csv`](results/tc_synth5k_serial.csv). |
| TC | Synth-5k | omp (vertex-parallel, dynamic) | 4 threads, degree+merge, `--validate` | 0.825 | — | Speedup ≈ **3.26×** vs serial. See [`results/tc_synth5k_omp_t4.csv`](results/tc_synth5k_omp_t4.csv). |
| TC | Synth-5k | omp_edge (edge-parallel + flatten) | 4 threads, degree+merge, `--validate` | 1.057 | — | Speedup ≈ **2.55×** vs serial; flatten cost loses to vertex-parallel on this graph. See [`results/tc_synth5k_omp_edge_t4.csv`](results/tc_synth5k_omp_edge_t4.csv). |

Sample-graph check (hand distances + tree/relaxation validate): [`results/bfs_sample_serial_cli.csv`](results/bfs_sample_serial_cli.csv).

Sample omp BFS check (distances vs serial, 1–N threads): [`results/bfs_sample_omp.csv`](results/bfs_sample_omp.csv), CLI [`results/bfs_sample_omp_cli.csv`](results/bfs_sample_omp_cli.csv).

Sample omp_atomic BFS check: [`results/bfs_sample_omp_atomic.csv`](results/bfs_sample_omp_atomic.csv), CLI [`results/bfs_sample_omp_atomic_cli.csv`](results/bfs_sample_omp_atomic_cli.csv).

Sample mpi_atomic BFS check (1/2/4/8 ranks, including empty ranks): [`results/bfs_sample_mpi_atomic_np1.csv`](results/bfs_sample_mpi_atomic_np1.csv), [`results/bfs_sample_mpi_atomic_np2.csv`](results/bfs_sample_mpi_atomic_np2.csv), [`results/bfs_sample_mpi_atomic_np4.csv`](results/bfs_sample_mpi_atomic_np4.csv), [`results/bfs_sample_mpi_atomic_np8.csv`](results/bfs_sample_mpi_atomic_np8.csv). CLI (`mpiexec -n 2`): [`results/bfs_sample_mpi_atomic_cli.csv`](results/bfs_sample_mpi_atomic_cli.csv).

Sample CC check (2 components, largest=4): [`results/cc_sample_serial_cli.csv`](results/cc_sample_serial_cli.csv), [`results/cc_sample_serial.csv`](results/cc_sample_serial.csv).

Sample TC check (6 triangles): [`results/tc_sample_serial.csv`](results/tc_sample_serial.csv), [`results/tc_sample_omp.csv`](results/tc_sample_omp.csv), [`results/tc_sample_omp_edge.csv`](results/tc_sample_omp_edge.csv); CLI [`results/tc_sample_serial_cli.csv`](results/tc_sample_serial_cli.csv). Design notes: [`docs/tc-parallel-design.md`](docs/tc-parallel-design.md).

**Speedup over serial baseline (higher is better)**

| Kernel / Graph | omp | omp_atomic / omp_edge | simd | cuda | mpi (N ranks) |
|---|---|---|---|---|---|
| BFS / LiveJournal | 0.81× | 7.14× (omp_atomic) | | | 5.03× (2 ranks × 7 threads) |
| BFS / USA road | | | | | |
| PageRank / LiveJournal | | | | | |
| PageRank / Kronecker-22 | | | | | |
| CC / Orkut | | | | | |
| CC / USA road | | | | | |
| TC / Synth-5k (4 threads) | 3.26× | 2.55× (omp_edge) | | | |
| TC / Orkut | | | | | |
| TC / Kronecker-22 | | | | | |

**Figures to produce** (all via `scripts/plot_results.py`)

- Strong scaling: speedup and parallel efficiency vs. threads, per kernel
- Weak scaling on Kronecker graphs
- BFS per-level time and frontier size, with the direction switch marked
- Power-law vs. road network: where the best strategy flips, per kernel
- Achieved memory bandwidth vs. hardware peak (roofline-style) for each kernel
- Triangle counting: scalar vs. SIMD intersection across degree distributions
- Effect of vertex reordering (none / degree / RCM / random)
- MPI: compute vs. communication share as ranks increase
- GPU: load-balancing strategy comparison (thread / warp / block per vertex)

## Design Notes

**Graph representation.** CSR with 32-bit vertex IDs and 64-bit offsets when edge count exceeds 2³². Neighbor lists sorted. Separate in-edge CSR for pull-based PageRank.

**BFS.** Level-synchronous. Discovery with a plain check then CAS; thread-local buffers merged by prefix sum. Direction optimization (Beamer et al.) switches to bottom-up with a bitmap frontier when the frontier is large; bottom-up needs no atomics.

**PageRank.** Pull-based formulation: each vertex sums contributions from its in-neighbors, so each output is written by one thread and no atomics are needed. Memory-bandwidth-bound; compare against the bandwidth roofline. Handle dangling vertices explicitly.

**Connected Components.** Compare label propagation, Shiloach-Vishkin, and Afforest (sampling-based). Use atomic CAS or `atomicMin` for hooking; pointer jumping to shorten chains. Report rounds to convergence.

**Triangle Counting.** Orient edges from lower to higher degree (or ID) so each triangle is counted once and hub work shrinks. Intersect sorted adjacency lists with merge or galloping (SIMD/bitmap later). Two OpenMP designs: vertex-parallel (`omp`) and edge-parallel (`omp_edge`); see [`docs/tc-parallel-design.md`](docs/tc-parallel-design.md). Schedule dynamically because work per edge varies enormously.

**SIMD scope.** Most useful for triangle counting (set intersection) and bitmap operations; modest for BFS and PageRank because they are memory-bound. Report both wins and non-wins.

**CUDA.** Degree-based work assignment (thread / warp / block per vertex), coalescing-friendly layouts, warp-level primitives (shuffle, ballot). Road networks expose kernel-launch overhead from thousands of small BFS levels.

**MPI.** 1D vertex partitioning first (hybrid MPI + OpenMP, one rank per node); 2D partitioning is a stretch goal. Overlap communication with computation where possible and measure the communication share of runtime.

More detail and profiler analysis go in [`docs/writeup.md`](docs/writeup.md).
