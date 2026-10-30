# Parallel BFS designs

Level-synchronous BFS across the variants in this repository. Distances are
validated against `bfs_serial`; parents are any valid same-level parent
(Graph500 tree checks).

## Variants

| Variant | File | Parallelism | Discovery | Notes |
|---|---|---|---|---|
| `serial` | `src/bfs/serial.cpp` | none | queue | Correctness oracle |
| `omp` | `src/bfs/omp.cpp` | OpenMP | no atomics; thread-local candidates → reduce-merge → sort/dedupe → claim | Deterministic parent (min id); slow on power-law graphs |
| `omp_atomic` | `src/bfs/omp_atomic.cpp` | OpenMP | atomic TAS on `visited[]`; thread-local next frontier | Default shared-memory workhorse |
| `omp_diropt` | `src/bfs/omp_diropt.cpp` | OpenMP | top-down TAS + bottom-up bitmap (Beamer / GAPBS) | Direction switch via `--alpha` / `--beta` |
| `mpi_atomic` | `src/bfs/mpi_atomic.cpp` | MPI + OpenMP | 1D vertex ownership; atomic local TAS; alltoallv discoveries | Hybrid; compute/comm split in results |

SIMD and CUDA BFS are roadmap items (Phases 3–4); not in this change set.

## Design A — `omp` (no-atomic + merge)

**Idea.** Parallelize over the current frontier. Each thread appends
`(neighbor, parent)` candidates into a private buffer **without** marking
visited. After the parallel region, fold buffers with a binary reduction tree,
sort/dedupe by `(vertex, parent)`, then claim unique vertices.

**Pros.** No atomics on the hot path; deterministic parents.

**Cons.** Sort/dedupe of a large multiset of candidates dominates on social /
power-law graphs (LiveJournal: ~0.81× vs serial in committed results). Many
duplicate discoveries for hub neighbors.

**When it wins.** Graphs where frontiers stay small and duplicates are rare
(not the common case for analytics graphs).

## Design B — `omp_atomic` (atomic test-and-set)

**Idea.** Same level-synchronous outer loop. For each neighbor, plain-load then
`compare_exchange` on a `visited[]` byte. The winning thread alone writes
`dist`/`parent` and pushes into a thread-local next frontier; frontiers merge
via the same reduction tree.

**Pros.** Claims once; no global sort. Strong scaling on mid-size social graphs
(LiveJournal: ~7.1× vs serial @ 14 threads in committed results).

**Cons.** Atomic contention on high-degree hubs; still top-down only, so mid-BFS
levels that touch a huge fraction of edges pay for scanning them from the
frontier side.

## Design C — `omp_diropt` (direction-optimizing) — **implemented here**

**Idea.** Beamer et al. / GAPBS hybrid:

1. **Top-down** (same as Design B) while the frontier’s out-edge volume
   (`scout_count`) is modest.
2. Switch to **bottom-up** when `scout_count > edges_to_check / alpha`
   (and `edges_to_check >= alpha`, so tiny graphs do not spuriously switch).
3. **Bottom-up:** every *unvisited* vertex scans **in-neighbors** against a
   bitmap of the current frontier. First hit is a valid parent; no atomics on
   the claim (one OpenMP iteration owns each `v`).
4. Switch back to top-down when frontier size `< n / beta`.

Defaults: `alpha=15`, `beta=18` (GAPBS). CLI: `--alpha`, `--beta`, `--per-level`.

**Directed graphs.** Bottom-up needs in-edges. For `directed=true` CSRs the
kernel builds a reverse CSR once per call; undirected/symmetrized CSRs reuse
the forward adjacency.

**Bitmap.** `include/bitmap.hpp` — dense 64-bit words for frontier membership.

**Pros.** Avoids scanning huge frontiers’ full adjacency when most vertices are
already reached; bottom-up needs no discovery atomics. Helps on social-like /
preferential-attachment structure.

**Cons.** Reverse CSR cost on directed inputs; bottom-up scans all unvisited
vertices (wasteful on road networks with thousands of tiny levels); TEPS is
not comparable to top-down-only variants because `edges_examined` counts
different work.

## Design D — `mpi_atomic` (hybrid 1D)

**Idea.** Contiguous 1D vertex ownership. Each rank expands its local frontier
with OpenMP + atomic TAS; remote discoveries are filtered to one send per
vertex and exchanged; owners accept with the same atomic. Full `dist`/`parent`
assembled on every rank.

**Pros.** Multi-node / multi-process scaling; reports compute vs communication.

**Cons.** Replicated full CSR today; communication share grows with ranks;
parents are any valid same-level parent (not deterministic across ranks).

## Alternatives considered (not shipped)

| Alternative | Why not (yet) |
|---|---|
| Prefix-sum / bag-of-tasks frontier (Ligra style) | Larger runtime rewrite; Designs B/C already match project interfaces |
| Always-bottom-up | Loses on road-like graphs; heuristic switching is the point of diropt |
| Bitmap frontier for top-down only | Extra fill cost without the bottom-up payoff |
| 2D MPI partitioning | Stretch goal in README; 1D is the Phase 5 baseline |
| CUDA BFS | Phase 4; separate from this OpenMP/MPI track |

## Correctness

- `validate_bfs`: Graph500 parent/level checks + edge relaxation.
- Parallel vs serial: exact `dist` and `visited`. `edges_examined` must match
  for top-down-only variants; **not** required for `omp_diropt`.
- Unit tests: `tests/test_bfs_{serial,omp,omp_atomic,omp_diropt,mpi_atomic}.cpp`.

## Benchmarks

### Committed LiveJournal (Windows host, prior commits)

See README Results. Summary vs serial median:

| Variant | Speedup | Notes |
|---|---|---|
| `omp` | 0.81× | Candidate sort/dedupe bound |
| `omp_atomic` | 7.14× | 14 threads |
| `mpi_atomic` | 5.03× | 2 ranks × 7 threads |

### Synthetic microbench (`tests/bench_bfs_parallel.cpp`)

Cloud agent Linux, Clang Release, 4 OpenMP threads, 5 sources × 3 trials.
CSVs under `results/{grid256,ba50k,bfs_sample}_bfs_<variant>.csv`.

| Graph | Structure | serial ms | omp ms | omp_atomic ms | omp_diropt ms |
|---|---|---|---|---|---|
| `grid256` (65K verts) | road-like, many levels | **0.41** | 9.77 | 5.21 | 5.68 |
| `ba50k` (50K verts, BA) | social-like, skewed | 1.27 | 7.87 | 1.16 | **0.62** |

Takeaway: on this machine size, serial wins on the grid (parallel fork/join and
atomics dominate tiny levels). On the BA graph, `omp_diropt` is the fastest
shared-memory variant; `omp` lags everywhere due to sort/dedupe.

Re-run:

```bash
cmake --build build --target bench_bfs_parallel
./build/bench_bfs_parallel
```

### CLI

```bash
./build/pargraph --kernel bfs --variant omp_diropt \
  --graph data/livejournal.csr --sources 64 --threads 14 \
  --alpha 15 --beta 18 --validate --per-level \
  --output results/lj_bfs_omp_diropt.csv
```

## Mapping to research question

*Which strategy wins on which structure?*

- **Power-law / social:** prefer `omp_atomic` or `omp_diropt`; avoid plain `omp`.
- **High-diameter / road:** serial or carefully tuned top-down; diropt rarely
  helps when levels stay tiny (and may hurt).
- **Multi-socket / multi-node:** `mpi_atomic` once the graph exceeds one NUMA
  domain; watch `comm_share` in the meta JSON.
