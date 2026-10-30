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

struct BfsResult {
  std::vector<std::int32_t> dist;    // kBfsUnreached if not reached
  std::vector<std::int32_t> parent;  // -1 if unreached; source parents itself
  std::uint64_t visited = 0;
  std::uint64_t edges_examined = 0;  // sum of degrees of dequeued vertices
  MpiPhaseStats mpi;
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

// Connected components via serial union-find (path compression + union by rank).
// Treats each stored arc u->v as an undirected link (weakly connected components
// on directed CSRs; full CC on undirected/symmetrized CSRs).
struct CcResult {
  std::vector<vtx_t> label;  // canonical root id per vertex (after compress)
  std::uint64_t num_components = 0;
  std::uint64_t largest_component = 0;
  std::uint64_t edges_united = 0;  // arcs processed in the union pass
};

struct CcValidation {
  bool ok = true;
  std::vector<std::string> errors;
};

CcResult cc_serial_unionfind(const Graph& g);

// Partition checks: every edge endpoints share a label; sizes sum to V;
// num_components / largest_component match the label multiset.
CcValidation validate_cc(const Graph& g, const CcResult& r);

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

}  // namespace pargraph
