# PageRank: parallel designs

PageRank follows the same layout as BFS: `src/pagerank/{serial,omp,omp_atomic,mpi}.cpp`,
per-variant tests under `tests/`, CLI via `pargraph --kernel pagerank`, and CSV/meta
under `results/`. Reverse (in-edge) CSR is built once in the driver and excluded from
kernel time.

Defaults: `--damping 0.85 --tolerance 1e-6 --max-iters 100`.

## Implemented designs

### 1. Serial pull (`--variant serial`)

Power iteration over the **in-CSR**. Each vertex `v` reads in-neighbors and writes
only `next[v]`:

```
dangling = Σ { score[u] | outdeg(u) = 0 }
base     = (1 − d)/N + d · dangling / N
next[v]  = base + d · Σ_{u ∈ in(v)} score[u] / outdeg(u)
```

Stop when `‖next − score‖₁ < tolerance` or `max_iters` is hit. Dangling mass is
redistributed uniformly every iteration.

### 2. OpenMP pull (`--variant omp`) — primary shared-memory design

Same math as serial. Parallel `omp for` over vertices:

- dangling sum and L1 residual use OpenMP `reduction(+)`
- each thread writes only `next[v]` for its assigned vertices → **no atomics**
- `schedule(dynamic, 64)` on the pull loop for degree skew

This matches the project Design Notes (“pull so each output is written by one
thread”). Scores match serial to ~1e-9 MAE/L∞ on the sample graph.

### 3. OpenMP push + atomics (`--variant omp_atomic`) — alternative

Scatter formulation over the **out-CSR**:

```
next[v] ← base
for each u with outdeg(u) > 0:
  contrib = d · score[u] / outdeg(u)
  for v ∈ out(u): atomic_add(next[v], contrib)
```

Atomic adds use CAS on `std::atomic<double>`. Same dangling/base handling as pull.
Floating-point add order differs across threads, so `--validate` compares to serial
with a looser MAE/L∞ bound (`1e-3`).

### 4. MPI + OpenMP pull (`--variant mpi`, optional `-DPARGRAPH_ENABLE_MPI=ON`)

Reuses BFS `mpi_partition_vertices_1d` (1D contiguous vertex ownership, replicated
CSR). Each rank:

1. reduces local dangling → `MPI_Allreduce(SUM)`
2. pull-updates owned vertices with OpenMP
3. `MPI_Allgatherv` owned score slices into a full vector
4. `MPI_Allreduce` residual

Reports slowest-rank compute/comm split and bytes sent (BFS-style). Not benchmarked
in this environment (MPI CMake path targets MS-MPI on Windows).

## Alternatives considered (not implemented)

| Idea | Why not chosen |
|---|---|
| Push without atomics (thread-local next + merge) | Extra `O(threads · V)` memory and a merge pass every iteration; atomics already provide a clean contrast to pull |
| Gauss–Seidel / async PageRank | Harder to validate against serial power iteration; different convergence semantics |
| Vertex reordering / blocked CSR | Orthogonal optimization; deferred with the global reordering roadmap |
| 2D / edge partitioning for MPI | Stretch goal in README; 1D + Allgatherv matches current BFS infra |

## Correctness

- Scores finite, non-negative, sum ≈ 1
- Parallel vs serial: MAE and L∞ within tolerance (pull: tight; push: `1e-3`)
- Hand check on `tests/tiny_graphs/pagerank_sample.el` (5-node digraph) and uniform
  3-cycle (`1/3` each)

## Benchmarks (this PR)

**Machine:** Linux cloud agent, 4 cores, Release g++/OpenMP.  
**Graph:** synthetic directed preferential-ish `data/synth_pa50k.csr` (V=50,000, E=500,000; regenerate via edgelist + `convert_graph`).  
**Setup:** damping=0.85, tolerance=1e-6, max_iters=100, 1 warmup + 5 trials, `--validate`. Reverse-CSR build excluded from kernel time (~44 ms).

| Variant | Threads | Median kernel (ms) | Median ms/iter | Iters | Speedup vs serial |
|---|---|---|---|---|---|
| serial (pull) | 1 | 9.13 | 0.761 | 12 | 1.00× |
| omp (pull) | 4 | 6.10 | 0.508 | 12 | **1.50×** |
| omp_atomic (push) | 4 | 25.32 | 2.110 | 12 | 0.36× |

Raw rows: [`results/synth_pa50k_pagerank_serial.csv`](../results/synth_pa50k_pagerank_serial.csv),
[`results/synth_pa50k_pagerank_omp.csv`](../results/synth_pa50k_pagerank_omp.csv),
[`results/synth_pa50k_pagerank_omp_atomic.csv`](../results/synth_pa50k_pagerank_omp_atomic.csv)
(+ `.meta.json` siblings).

**Takeaway:** On this memory-bound sweep, **pull OpenMP wins**; push loses to atomic
contention on high-degree destinations. That is the intended design contrast (same
spirit as BFS `omp` vs `omp_atomic`, with the opposite winner because PageRank has
no frontier discovery phase).

Sample CLI checks: `results/pagerank_sample_*_cli.csv`.
