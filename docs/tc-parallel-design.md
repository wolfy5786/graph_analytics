# Triangle Counting — Parallel Designs

This note documents the forward triangle-counting kernel and the OpenMP
designs implemented alongside the serial baseline. It mirrors the BFS pattern
in this repo: multiple competing shared-memory designs, validated against the
serial oracle, with sample benches under `results/`.

## Algorithm (serial baseline)

**Forward counting with degree (or id) orientation**

1. Treat the input CSR as undirected (symmetrized adjacency, sorted lists).
2. **Orient** each undirected edge `{u,v}` as `u → v` iff
   - `degree`: `(deg(u), u) < (deg(v), v)`, or
   - `id`: `u < v`.
   Orientation is preprocessing (`orient_time_ms`), excluded from kernel time.
3. For each oriented edge `u → v`, add `|N⁺(u) ∩ N⁺(v)|` where `N⁺` is the
   forward (oriented) adjacency.
4. Each undirected triangle is counted **exactly once** (the lowest-ranked
   vertex is the apex of the oriented wedge).

**Intersection modes** (`--intersect`)

| Mode | When it helps |
|---|---|
| `merge` (default) | Balanced degree pairs; linear scan of both sorted lists |
| `galloping` | Skewed pairs; exponential probe + binary search on the longer list |

SIMD / bitmap intersections are roadmap Phase 3 and intentionally out of scope
here.

## Parallel designs implemented

Both designs run on a **pre-oriented** forward CSR and use OpenMP
`reduction(+:triangles)` — triangle counting needs no atomics for the total.

### Design A — `omp` (vertex-parallel)

```text
#pragma omp parallel for schedule(dynamic, 64) reduction(+:triangles)
for u in V:
  for v in N+(u):
    triangles += |N+(u) ∩ N+(v)|
```

- **Granularity:** one vertex’s entire out-neighborhood.
- **Scheduling:** `dynamic, 64` to absorb residual degree skew after orientation.
- **Pros:** no edge-list materialization; simple; good when out-degrees are
  moderate after orientation.
- **Cons:** a single hub with large forward degree still serializes a big
  chunk of work onto one OpenMP iteration.

### Design B — `omp_edge` (edge-parallel)

```text
edges = flatten(N+)                         # O(E+) once per kernel call
#pragma omp parallel for schedule(dynamic, 64) reduction(+:triangles)
for (u, v) in edges:
  triangles += |N+(u) ∩ N+(v)|
```

- **Granularity:** one oriented edge (one intersection).
- **Pros:** finer load balance when a few vertices dominate `N⁺`.
- **Cons:** pays `O(E+)` flatten + extra memory traffic every timed trial
  (included in kernel time so the comparison is honest).

## Alternatives considered (not implemented)

| Alternative | Why deferred |
|---|---|
| Node-iterator (enumerate wedges in `N(u)`, adjacency probe) | Worse on power-law graphs; orientation + edge iterator is the GAP/GraphChallenge default |
| Atomic per-vertex triangle tallies | Unnecessary for a global count; reduction is enough |
| MPI 1D edge / vertex partition | Follows BFS later; TC communication of adjacency cuts is a separate design |
| CUDA warp-per-edge intersection | Phase 4 |
| SIMD intersect (`intersect.hpp` AVX path) | Phase 3; scalar merge/galloping first |

## Correctness

`validate_tc`:

1. Serial recount with the same `(orient, intersect)`.
2. Cross-check merge ↔ galloping totals.
3. On tiny graphs (`V≤64`, `E≤512`), brute undirected edge-intersection `/ 3`.

Parallel variants must match the serial total exactly (integer result).

## CLI / layout (BFS-style)

| Piece | Path |
|---|---|
| Interfaces | `include/kernels.hpp`, `include/intersect.hpp` |
| Serial | `src/tc/serial.cpp` |
| Design A | `src/tc/omp.cpp` |
| Design B | `src/tc/omp_edge.cpp` |
| Tests | `tests/test_tc_{serial,omp,omp_edge}.cpp` |
| Tiny graph | `tests/tiny_graphs/tc_sample.el` (6 triangles) |
| Sample benches | `results/tc_sample_{serial,omp,omp_edge}.csv` (+ `.meta.json`) |

```bash
./build/pargraph --kernel tc --variant serial --graph tests/tiny_graphs/tc_sample.csr \
  --orient degree --intersect merge --validate --trials 3 \
  --output results/tc_sample_serial_cli.csv

./build/pargraph --kernel tc --variant omp --threads 4 --graph ... --validate
./build/pargraph --kernel tc --variant omp_edge --threads 4 --graph ... --validate
```

## Benchmark protocol

- Warm-up + median over timed trials; report `intersect_rate = intersect_ops / s`.
- Keep `orient_time_ms` and load time out of the kernel median.
- Compare **serial vs omp vs omp_edge** on the same oriented graph, same
  intersect mode, same thread sweep (`1,2,4,…`).
- Prefer a mid-size undirected graph (e.g. LiveJournal symmetrized, Orkut) for
  speedup tables; tiny `tc_sample` only proves wiring.

## Expected qualitative behavior

- Orientation by **degree** usually shrinks hub work vs id orientation.
- **`omp_edge`** should pull ahead of **`omp`** when forward-degree skew remains
  high; on flat graphs the flatten cost can make **`omp`** win.
- **Galloping** helps high-skew intersections; **merge** wins on similar-length
  lists.

## Measured on Synth-5k (this PR)

Machine: Linux cloud agent · GCC 13 · Release · OpenMP. Graph:
`tests/tiny_graphs/tc_synth_5k.csr` (V=5000, E=89658 undirected, 6924 triangles).
Degree orientation + merge unless noted. Kernel medians exclude orient/load.

| Variant | Threads | Median kernel (ms) | Speedup vs serial | Artifact |
|---|---:|---:|---:|---|
| serial | 1 | 2.691 | 1.00× | `results/tc_synth5k_serial.csv` |
| omp | 1 | 2.796 | 0.96× | `results/tc_synth5k_omp_t1.csv` |
| omp | 2 | 1.474 | 1.83× | `results/tc_synth5k_omp_t2.csv` |
| omp | 4 | 0.825 | **3.26×** | `results/tc_synth5k_omp_t4.csv` |
| omp_edge | 1 | 3.183 | 0.85× | `results/tc_synth5k_omp_edge_t1.csv` |
| omp_edge | 2 | 1.765 | 1.52× | `results/tc_synth5k_omp_edge_t2.csv` |
| omp_edge | 4 | 1.057 | 2.55× | `results/tc_synth5k_omp_edge_t4.csv` |
| omp + galloping | 4 | 1.028 | 2.62× | `results/tc_synth5k_omp_galloping_t4.csv` |
| omp + id orient | 4 | 0.961 | 2.80× | `results/tc_synth5k_omp_id_t4.csv` |

**Takeaway on this graph:** vertex-parallel `omp` beats `omp_edge` at every
thread count (flatten overhead). Degree+merge at 4 threads is the best of the
measured set; galloping and id-orient trail slightly here (low residual skew).
