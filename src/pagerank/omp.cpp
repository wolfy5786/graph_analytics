#include "kernels.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include <omp.h>

namespace pargraph {

PageRankResult pagerank_omp(const Graph& out_g, const Graph& in_g,
                            PageRankParams params, int num_threads) {
  PageRankResult r;
  const offset_t n = out_g.num_vertices;
  r.scores.assign(static_cast<std::size_t>(n), 0.0);
  if (n == 0 || in_g.num_vertices != n) {
    return r;
  }
  if (params.max_iters < 1) {
    params.max_iters = 1;
  }
  if (!(params.damping >= 0.0 && params.damping <= 1.0) ||
      !std::isfinite(params.damping)) {
    params.damping = 0.85;
  }
  if (!(params.tolerance > 0.0) || !std::isfinite(params.tolerance)) {
    params.tolerance = 1e-6;
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  const double inv_n = 1.0 / static_cast<double>(n);
  const double d = params.damping;
  std::vector<double> curr(static_cast<std::size_t>(n), inv_n);
  std::vector<double> next(static_cast<std::size_t>(n), 0.0);

  for (int it = 0; it < params.max_iters; ++it) {
    double dangling = 0.0;
#pragma omp parallel for reduction(+ : dangling) schedule(static)
    for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
      if (out_g.degree(static_cast<vtx_t>(u)) == 0) {
        dangling += curr[static_cast<std::size_t>(u)];
      }
    }

    const double base = (1.0 - d) * inv_n + d * dangling * inv_n;
    double residual = 0.0;

    // Pull: each vertex writes only next[v] — no atomics, no races.
#pragma omp parallel for reduction(+ : residual) schedule(dynamic, 64)
    for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
      double sum = 0.0;
      const auto* begin = in_g.neighbors_begin(static_cast<vtx_t>(v));
      const auto* end = in_g.neighbors_end(static_cast<vtx_t>(v));
      for (const auto* itn = begin; itn != end; ++itn) {
        const vtx_t u = *itn;
        const offset_t deg = out_g.degree(u);
        sum += curr[static_cast<std::size_t>(u)] / static_cast<double>(deg);
      }
      const double nv = base + d * sum;
      residual += std::fabs(nv - curr[static_cast<std::size_t>(v)]);
      next[static_cast<std::size_t>(v)] = nv;
    }

    curr.swap(next);
    r.iterations = it + 1;
    r.residual_l1 = residual;
    r.edges_examined += out_g.num_edges;
    if (residual < params.tolerance) {
      break;
    }
  }

  r.scores = std::move(curr);
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    sum += r.scores[static_cast<std::size_t>(i)];
  }
  r.score_sum = sum;
  return r;
}

}  // namespace pargraph
