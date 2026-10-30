#include "intersect.hpp"
#include "kernels.hpp"

#include <cstdint>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

// Flatten oriented CSR into an edge list (u, v) so the parallel loop iterates
// edges rather than vertices. Built once per call; cost is O(E+) and is part
// of kernel time for an apples-to-apples design comparison against vertex-
// parallel (which does not build an edge list).
struct EdgeList {
  std::vector<vtx_t> u;
  std::vector<vtx_t> v;
};

EdgeList flatten_oriented(const Graph& oriented) {
  EdgeList el;
  el.u.resize(static_cast<std::size_t>(oriented.num_edges));
  el.v.resize(static_cast<std::size_t>(oriented.num_edges));
  std::size_t w = 0;
  for (offset_t u = 0; u < oriented.num_vertices; ++u) {
    const auto* begin = oriented.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = oriented.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      el.u[w] = static_cast<vtx_t>(u);
      el.v[w] = *it;
      ++w;
    }
  }
  return el;
}

}  // namespace

// Design B — edge-parallel OpenMP TC:
// Materialize oriented edges, then parallel-for over edges with a dynamic
// schedule. Better load balance when a few hubs dominate out-degree after
// orientation; pays for an O(E) flatten before intersecting.
TcResult tc_omp_edge(const Graph& oriented, IntersectMode mode,
                     int num_threads) {
  TcResult r;
  if (oriented.num_vertices == 0 || oriented.num_edges == 0) {
    return r;
  }
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  const EdgeList edges = flatten_oriented(oriented);
  const std::int64_t m = static_cast<std::int64_t>(edges.u.size());

  std::uint64_t triangles = 0;
  std::uint64_t wedges = 0;
  std::uint64_t ops = 0;

#pragma omp parallel for reduction(+ : triangles, wedges, ops) \
    schedule(dynamic, 64)
  for (std::int64_t e = 0; e < m; ++e) {
    const vtx_t u = edges.u[static_cast<std::size_t>(e)];
    const vtx_t v = edges.v[static_cast<std::size_t>(e)];
    ++wedges;
    ++ops;
    triangles += intersect_sorted(
        oriented.neighbors_begin(u), oriented.neighbors_end(u),
        oriented.neighbors_begin(v), oriented.neighbors_end(v), mode);
  }

  r.triangles = triangles;
  r.wedges_examined = wedges;
  r.intersect_ops = ops;
  return r;
}

}  // namespace pargraph
