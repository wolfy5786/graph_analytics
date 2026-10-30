#pragma once

#include "graph.hpp"
#include "intersect.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pargraph {

inline constexpr std::int32_t kBfsUnreached =
    std::numeric_limits<std::int32_t>::max();

// Phase split for the hybrid MPI BFS. Zero for shared-memory variants.
// compute/comm are taken from the slowest rank; comm includes time spent
// waiting inside MPI. bytes_sent is the sum of discovery payloads.
struct MpiPhaseStats {
  double compute_sec = 0;
  double comm_sec = 0;
  std::uint64_t bytes_sent = 0;
};

// Optional per-level timing for direction-optimizing / instrumented BFS.
struct BfsLevelStat {
  std::int32_t level = 0;
  std::uint64_t frontier = 0;
  std::uint64_t edges_examined = 0;
  double time_ms = 0.0;
  bool bottom_up = false;
};

struct BfsResult {
  std::vector<std::int32_t> dist;    // kBfsUnreached if not reached
  std::vector<std::int32_t> parent;  // -1 if unreached; source parents itself
  std::uint64_t visited = 0;
  std::uint64_t edges_examined = 0;  // sum of degrees of dequeued vertices
  MpiPhaseStats mpi;
  std::vector<BfsLevelStat> levels;  // populated when per_level=true
};

// Serial queue-based level-synchronous BFS. Works on whatever orientation the
// CSR stores (directed out-edges or undirected symmetrized).
BfsResult bfs_serial(const Graph& g, vtx_t source);

// OpenMP level-synchronous BFS: thread-local neighbor discovery (no atomics /
// no visited marking), parallel reduction-tree merge, dedupe, then claim.
// num_threads <= 0 leaves the OpenMP default (typically all cores).
BfsResult bfs_omp(const Graph& g, vtx_t source, int num_threads = 0);

// OpenMP level-synchronous BFS with an atomic visited[] array: parallel over
// the current level; atomic test-and-set claims each neighbor once; winners
// write dist/parent and push into a thread-local next frontier, then merge.
BfsResult bfs_omp_atomic(const Graph& g, vtx_t source, int num_threads = 0);

// Direction-optimizing OpenMP BFS (Beamer et al. / GAPBS).
// Top-down uses atomic TAS + thread-local next frontiers (same as omp_atomic).
// Bottom-up scans unvisited vertices against a bitmap frontier via in-edges
// (built on the fly for directed CSRs; reused for undirected). Switches with
// alpha (growing) / beta (shrinking). Defaults match GAPBS (15 / 18).
// When per_level is true, fills BfsResult::levels with frontier size, edges,
// time, and top-down vs bottom-up for each level.
BfsResult bfs_omp_diropt(const Graph& g, vtx_t source, int num_threads = 0,
                         int alpha = 15, int beta = 18,
                         bool per_level = false);

// 1D contiguous vertex ownership. `local` holds the out-edges of [lo, hi)
// with global neighbor ids. Built from an identical full graph on every rank
// (replicated CSR, partitioned edges and frontier).
struct MpiVertexPart {
  int rank = 0;
  int ranks = 1;
  vtx_t lo = 0;
  vtx_t hi = 0;
  vtx_t base = 0;  // num_vertices / ranks
  vtx_t rem = 0;   // num_vertices % ranks
  offset_t num_vertices = 0;
  Graph local;
};

#if defined(PARGRAPH_HAS_MPI)
MpiVertexPart mpi_partition_vertices_1d(const Graph& g);

// Hybrid MPI + OpenMP level-synchronous BFS.
// Owner rank expands its frontier; an atomic test-and-set on a per-rank
// seen[] array claims local neighbors and filters remote discoveries to one
// send per vertex. Owners accept remote (vertex, parent) pairs with the same
// atomic. Distances match serial; parents are any valid same-level parent.
// Full dist/parent are assembled on every rank. MPI must already be initialized.
// num_threads <= 0 leaves the OpenMP default.
BfsResult bfs_mpi_atomic(const MpiVertexPart& part, vtx_t source,
                         int num_threads = 0);
#endif

struct BfsValidation {
  bool ok = true;
  std::vector<std::string> errors;
};

// Graph500-style tree / level checks against dist/parent.
BfsValidation validate_bfs(const Graph& g, vtx_t source, const BfsResult& r);

// Connected components. Every variant treats each stored arc u->v as an
// undirected link (WCC on directed CSRs; full CC on undirected/symmetrized).
struct CcResult {
  std::vector<vtx_t> label;  // canonical root id per vertex (after compress)
  std::uint64_t num_components = 0;
  std::uint64_t largest_component = 0;
  std::uint64_t edges_united = 0;  // arcs examined / hooked
  std::uint64_t rounds = 0;        // iterations / hooking rounds to fixpoint
  MpiPhaseStats mpi;
};

struct CcValidation {
  bool ok = true;
  std::vector<std::string> errors;
};

// Serial union-find (path compression + union by rank). Correctness oracle.
CcResult cc_serial_unionfind(const Graph& g);

// OpenMP two-phase label propagation: each round computes next[v] =
// min(label[v], min neighbor labels) into a private buffer, then swaps.
// No atomics; barriers separate the read and write phases.
// num_threads <= 0 leaves the OpenMP default.
CcResult cc_omp(const Graph& g, int num_threads = 0);

// OpenMP CC with atomic parent updates. algo selects the hooking rule:
//   "labelprop" (default) — atomicMin of labels over endpoints each round
//   "sv"                  — Shiloach-Vishkin hook (CAS) + pointer jumping
//   "afforest"            — neighbor sampling then finish small components
CcResult cc_omp_atomic(const Graph& g, int num_threads = 0,
                       const std::string& algo = "labelprop");

#if defined(PARGRAPH_HAS_MPI)
// Hybrid MPI + OpenMP label propagation. Each rank owns a 1D vertex slice,
// expands its owned edges with OpenMP atomicMin, then Allreduce(MPI_MIN) on
// the full label array. Labels/partitions match serial WCC. MPI must already
// be initialized. num_threads <= 0 leaves the OpenMP default.
CcResult cc_mpi_atomic(const MpiVertexPart& part, int num_threads = 0);
#endif

// From canonical labels, set num_components / largest_component.
void cc_fill_aggregates(CcResult& r);

// Partition checks: every edge endpoints share a label; sizes sum to V;
// num_components / largest_component match the label multiset.
CcValidation validate_cc(const Graph& g, const CcResult& r);

// True iff a and b induce the same partition (label values may differ).
bool cc_same_partition(const std::vector<vtx_t>& a, const std::vector<vtx_t>& b);

// ---- Triangle counting ----------------------------------------------------
// Orient undirected (or symmetrized) edges from lower to higher rank so each
// triangle is counted once via forward-adjacency intersection.
enum class TcOrientMode : std::uint8_t { Degree = 0, Id = 1 };

struct TcResult {
  std::uint64_t triangles = 0;
  std::uint64_t wedges_examined = 0;  // oriented edges processed
  std::uint64_t intersect_ops = 0;    // intersection calls (= wedges)
};

struct TcValidation {
  bool ok = true;
  std::vector<std::string> errors;
};

// Build a directed forward CSR: keep u→v iff rank(u) < rank(v) under orient.
// Neighbor lists are sorted. Orientation is preprocessing, not kernel time.
Graph orient_graph(const Graph& g, TcOrientMode orient);

// Serial forward triangle counting on an undirected/symmetrized CSR.
TcResult tc_serial(const Graph& g, TcOrientMode orient = TcOrientMode::Degree,
                   IntersectMode mode = IntersectMode::Merge);

// Serial count on a pre-oriented forward CSR (kernel-only path).
TcResult tc_serial_on_oriented(const Graph& oriented,
                               IntersectMode mode = IntersectMode::Merge);

// OpenMP Design A: parallel over vertices, dynamic schedule, reduction.
TcResult tc_omp(const Graph& oriented, IntersectMode mode = IntersectMode::Merge,
                int num_threads = 0);

// OpenMP Design B: flatten oriented edges, parallel over edges, reduction.
TcResult tc_omp_edge(const Graph& oriented,
                     IntersectMode mode = IntersectMode::Merge,
                     int num_threads = 0);

// Exact-total checks: serial recount, merge↔galloping cross-check, and on
// tiny graphs a brute edge-intersection / 3 oracle (needs symmetrized CSR).
TcValidation validate_tc(const Graph& g, const TcResult& r,
                         TcOrientMode orient = TcOrientMode::Degree,
                         IntersectMode mode = IntersectMode::Merge);

// ---- PageRank ---------------------------------------------------------------

struct PageRankParams {
  double damping = 0.85;
  double tolerance = 1e-6;  // L1 residual stop
  int max_iters = 100;
};

struct PageRankResult {
  std::vector<double> scores;
  int iterations = 0;
  double residual_l1 = 0.0;
  double score_sum = 0.0;
  std::uint64_t edges_examined = 0;  // iterations * E (pull/push sweeps)
  MpiPhaseStats mpi;
};

struct PageRankValidation {
  bool ok = true;
  std::vector<std::string> errors;
};

// Serial pull PageRank. `out_g` is out-CSR; `in_g` is the reverse (in-CSR).
// Dangling mass is redistributed uniformly each iteration.
PageRankResult pagerank_serial(const Graph& out_g, const Graph& in_g,
                               PageRankParams params = {});

// OpenMP pull PageRank: parallel over vertices; each thread writes only its
// own score (no atomics). Dangling mass and L1 residual via reductions.
// num_threads <= 0 leaves the OpenMP default.
PageRankResult pagerank_omp(const Graph& out_g, const Graph& in_g,
                            PageRankParams params = {}, int num_threads = 0);

// OpenMP push PageRank: each vertex scatters score/outdeg along out-edges
// with atomic adds into a next[] buffer. Contrasts with pull (omp).
// Uses out-CSR only (no reverse graph).
PageRankResult pagerank_omp_atomic(const Graph& out_g,
                                   PageRankParams params = {},
                                   int num_threads = 0);

#if defined(PARGRAPH_HAS_MPI)
// Hybrid MPI + OpenMP pull PageRank on a 1D vertex partition.
// Replicated CSRs on every rank; each rank updates owned scores then
// Allgatherv. Compute/comm split follows the BFS mpi pattern.
PageRankResult pagerank_mpi(const MpiVertexPart& part, const Graph& out_g,
                            const Graph& in_g, PageRankParams params = {},
                            int num_threads = 0);
#endif

// Invariants: score size, finite values, sum ≈ 1, residual finite.
// If `ref` is non-null, also checks L1/L∞ vs reference within `score_tol`.
PageRankValidation validate_pagerank(const Graph& out_g,
                                     const PageRankResult& r,
                                     const PageRankParams& params,
                                     const PageRankResult* ref = nullptr,
                                     double score_tol = 1e-6);

}  // namespace pargraph
