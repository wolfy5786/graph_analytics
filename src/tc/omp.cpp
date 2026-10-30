#include "intersect.hpp"
#include "kernels.hpp"

#include <cstdint>

#include <omp.h>

namespace pargraph {

// Design A — vertex-parallel OpenMP TC:
// Parallelize the outer vertex loop. Each thread walks u's forward neighbors
// and intersects sorted adjacency lists; triangles accumulate via reduction.
// Dynamic schedule absorbs degree skew after orientation.
TcResult tc_omp(const Graph& oriented, IntersectMode mode, int num_threads) {
  TcResult r;
  if (oriented.num_vertices == 0) {
    return r;
  }
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  std::uint64_t triangles = 0;
  std::uint64_t wedges = 0;
  std::uint64_t ops = 0;
  const offset_t n = oriented.num_vertices;

#pragma omp parallel for reduction(+ : triangles, wedges, ops) \
    schedule(dynamic, 64)
  for (std::int64_t u64 = 0; u64 < static_cast<std::int64_t>(n); ++u64) {
    const vtx_t u = static_cast<vtx_t>(u64);
    const auto* ub = oriented.neighbors_begin(u);
    const auto* ue = oriented.neighbors_end(u);
    for (const auto* it = ub; it != ue; ++it) {
      const vtx_t v = *it;
      ++wedges;
      ++ops;
      triangles += intersect_sorted(ub, ue, oriented.neighbors_begin(v),
                                    oriented.neighbors_end(v), mode);
    }
  }

  r.triangles = triangles;
  r.wedges_examined = wedges;
  r.intersect_ops = ops;
  return r;
}

}  // namespace pargraph
