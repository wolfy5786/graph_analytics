#include "kernels.hpp"

#include <atomic>
#include <cmath>
#include <utility>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

// Atomic double add via CAS on the bit pattern (C++17 portable).
void atomic_add_double(std::atomic<double>& cell, double delta) {
  double cur = cell.load(std::memory_order_relaxed);
  while (!cell.compare_exchange_weak(cur, cur + delta, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
  }
}

}  // namespace

PageRankResult pagerank_omp_atomic(const Graph& out_g, PageRankParams params,
                                   int num_threads) {
  PageRankResult r;
  const offset_t n = out_g.num_vertices;
  r.scores.assign(static_cast<std::size_t>(n), 0.0);
  if (n == 0) {
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
  std::vector<std::atomic<double>> next(static_cast<std::size_t>(n));

  for (int it = 0; it < params.max_iters; ++it) {
    double dangling = 0.0;
#pragma omp parallel for reduction(+ : dangling) schedule(static)
    for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
      if (out_g.degree(static_cast<vtx_t>(u)) == 0) {
        dangling += curr[static_cast<std::size_t>(u)];
      }
    }

    const double base = (1.0 - d) * inv_n + d * dangling * inv_n;
#pragma omp parallel for schedule(static)
    for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
      next[static_cast<std::size_t>(v)].store(base, std::memory_order_relaxed);
    }

    // Push: scatter contrib along out-edges with atomic adds into next[].
#pragma omp parallel for schedule(dynamic, 64)
    for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
      const offset_t deg = out_g.degree(static_cast<vtx_t>(u));
      if (deg == 0) {
        continue;
      }
      const double contrib =
          d * curr[static_cast<std::size_t>(u)] / static_cast<double>(deg);
      const auto* begin = out_g.neighbors_begin(static_cast<vtx_t>(u));
      const auto* end = out_g.neighbors_end(static_cast<vtx_t>(u));
      for (const auto* itn = begin; itn != end; ++itn) {
        atomic_add_double(next[static_cast<std::size_t>(*itn)], contrib);
      }
    }

    double residual = 0.0;
#pragma omp parallel for reduction(+ : residual) schedule(static)
    for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
      const double nv = next[static_cast<std::size_t>(v)].load(
          std::memory_order_relaxed);
      residual += std::fabs(nv - curr[static_cast<std::size_t>(v)]);
      curr[static_cast<std::size_t>(v)] = nv;
    }

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
