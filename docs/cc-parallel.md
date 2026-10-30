# Connected Components — Parallel Designs

This note mirrors the BFS layout: one serial oracle, then `omp` / `omp_atomic` /
`mpi_atomic` variants with the same CLI, validation style, results CSV schema, and
sample + mid-size benches.

Every variant treats each stored CSR arc as an **undirected** link (WCC on
directed graphs). Labels may differ across algorithms; correctness is **partition
equality** vs. serial union-find (`cc_same_partition` + `validate_cc`).

## Implemented designs

### 1. Serial union-find (`--variant serial --algo unionfind`)

Path compression + union-by-rank over every arc. Correctness oracle. No round
count (`rounds = 0`).

### 2. OpenMP, no atomics (`--variant omp --algo labelprop`)

BFS-`omp` pattern applied to CC:

1. Threads scan out-arcs and emit **undirected hook proposals**
   `(u ← label[v], v ← label[u])` into thread-local buffers (plus identity).
2. Parallel reduction-tree merge of buffers.
3. Sort / keep **minimum** proposed label per vertex; write back.
4. Repeat until a round makes no change.

Undirected proposals are required so directed CSRs still compute WCC (pull-from-
out-neighbors alone is insufficient).

**Trade-off:** same class of overhead as BFS `omp` — candidate volume + sort /
dedupe dominate on denser graphs. Documented as a deliberate structural twin of
BFS `omp`, not the performance pick.

### 3. OpenMP + atomics (`--variant omp_atomic`)

Three algorithms share the atomic parent array; select with `--algo`:

| `--algo` | Rule | Notes |
|---|---|---|
| `labelprop` (default) | `atomicMin` on both endpoints of each arc each round, then pointer jump | Fewest instructions per edge; usually fastest parallel design here |
| `sv` | Shiloach–Vishkin hook (CAS larger root onto smaller) + compress | Classic PRAM CC; more CAS traffic |
| `afforest` | Sample first *k* neighbors, compress, then finish remaining neighbors | Sampling shortens chains early; finish pass kept **exact** (no large-component skip) so partitions match serial |

### 4. Hybrid MPI + OpenMP (`--variant mpi_atomic --algo labelprop`)

Same 1D vertex ownership as BFS `mpi_atomic` (`mpi_partition_vertices_1d`):

1. Each rank hooks along **owned** out-arcs with OpenMP `atomicMin`.
2. `MPI_Allreduce(MPI_MIN)` over the full label array.
3. Repeat until a global “changed” flag clears.

Reports slowest-rank `compute_ms` / `comm_ms` and Allreduce payload bytes
(excluded: CSR load and 1D slice time), matching BFS MPI accounting.

## Alternatives considered

| Idea | Why not primary |
|---|---|
| Pull-only two-phase LP (read neighbors → private `next[]`) | Incorrect WCC on directed CSR (misses reverse reachability through arcs) |
| Classic Afforest “skip large components” finish | Can miss bridges between two large components; dropped for exact serial match |
| Rem’s algorithm / lock-free UF | Overlaps SV/Afforest; higher implementation risk for little extra coverage |
| Pointer-jumping-only without hooking min | Does not connect components |
| Full graph gather each MPI round vs. edge exchange | Allreduce-MIN is simple and matches the replicated-CSR MPI BFS setup; edge exchange is a stretch optimization |

## CLI

```bash
./build/pargraph --kernel cc --variant serial|omp|omp_atomic|mpi_atomic \
  --graph data/graph.csr --trials 3 --threads N --validate \
  --algo unionfind|labelprop|sv|afforest \
  --output results/out.csv
```

Defaults: `serial` → `unionfind`; parallel variants → `labelprop`.
`omp` only accepts `labelprop`. `sv` / `afforest` require `omp_atomic`.

## Validation

- `validate_cc`: every arc’s endpoints share a label; labels canonical; size
  aggregates match.
- Parallel runs also require `cc_same_partition` vs. serial union-find and equal
  `num_components` / `largest_component`.

## Benchmarks (this PR)

Machine: Linux cloud agent, 4 cores, Release + OpenMPI. Kernel time excludes CSR
load (and MPI 1D slice). 1 warmup + 3 timed trials, `--validate`.

### wiki-Vote (`data/wiki_vote.csr`, directed WCC)

V=8298, E=103689 → **1207** components, largest **7066**.

| Variant | Algo | Median ms | Rounds | Speedup vs serial |
|---|---|---:|---:|---:|
| serial | unionfind | 0.335 | 0 | 1.00× |
| omp | labelprop | 61.4 | 6 | 0.005× |
| omp_atomic | labelprop | 0.302 | 3 | **1.11×** |
| omp_atomic | sv | 1.33 | 2 | 0.25× |
| omp_atomic | afforest | 1.30 | 2 | 0.26× |
| mpi_atomic (2×2 thr) | labelprop | 0.703 | 4 | 0.48× |

Artifacts: `results/wiki_cc_*.csv` (+ `.meta.json`).

### Synthetic blocks (`data/synthetic_cc.csr`, undirected)

V=50000, E=798348 → **6** components, largest **29997**.

| Variant | Algo | Median ms | Rounds | Speedup vs serial |
|---|---|---:|---:|---:|
| serial | unionfind | 2.84 | 0 | 1.00× |
| omp | labelprop | 775 | 8 | 0.004× |
| omp_atomic | labelprop | 2.91 | 3 | 0.97× |
| omp_atomic | sv | 9.20 | 2 | 0.31× |
| omp_atomic | afforest | 5.86 | 2 | 0.48× |
| mpi_atomic (2×2 thr) | labelprop | 4.95 | 4 | 0.57× |

Artifacts: `results/synth_cc_*.csv` (+ `.meta.json`).

### Sample graph

`tests/tiny_graphs/cc_sample.el` (2 components, largest 4): unit tests write
`results/cc_sample_{serial,omp,omp_atomic,omp_atomic_sv,omp_atomic_afforest}.csv`
and MPI `results/cc_sample_mpi_atomic_np{1,2,4,8}.csv`. CLI mirrors under
`results/cc_sample_*_cli.csv`.

## Takeaways

1. **`omp_atomic` + `labelprop` (`atomicMin`)** is the winning shared-memory
   parallel design on these graphs — same role `omp_atomic` plays for BFS.
2. **`omp` (no-atomic merge)** is a structural twin of BFS `omp` and a clear
   non-win once candidate sort/dedupe dominates (again like BFS on LiveJournal).
3. **SV** and **Afforest** are correct and useful for the design comparison; both
   lose to plain `atomicMin` LP here (more CAS / extra passes).
4. **MPI** pays Allreduce cost on small/mid graphs; comm share ~20–33% of the
   compute+comm split on these inputs. Worth revisiting on LiveJournal-scale CSR
   with more ranks once that dataset is local.
5. Serial union-find remains extremely strong on mid-size inputs; parallel value
   shows up when UF’s random parent chasing loses to bandwidth-friendly label
   sweeps on larger power-law graphs (see existing `results/lj_cc_serial.csv`).
