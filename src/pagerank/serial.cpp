#include "kernels.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace pargraph {

PageRankResult pagerank_serial(const Graph& out_g, const Graph& in_g,
                               PageRankParams params) {
  PageRankResult r;
  const offset_t n = out_g.num_vertices;
  r.scores.assign(static_cast<std::size_t>(n), 0.0);
  if (n == 0) {
    return r;
  }
  if (in_g.num_vertices != n) {
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

  const double inv_n = 1.0 / static_cast<double>(n);
  const double d = params.damping;
  std::vector<double> curr(static_cast<std::size_t>(n), inv_n);
  std::vector<double> next(static_cast<std::size_t>(n), 0.0);

  for (int it = 0; it < params.max_iters; ++it) {
    double dangling = 0.0;
    for (offset_t u = 0; u < n; ++u) {
      if (out_g.degree(static_cast<vtx_t>(u)) == 0) {
        dangling += curr[static_cast<std::size_t>(u)];
      }
    }

    const double base = (1.0 - d) * inv_n + d * dangling * inv_n;
    double residual = 0.0;

    for (offset_t v = 0; v < n; ++v) {
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
  for (double s : r.scores) {
    r.score_sum += s;
  }
  return r;
}

PageRankValidation validate_pagerank(const Graph& out_g,
                                     const PageRankResult& r,
                                     const PageRankParams& params,
                                     const PageRankResult* ref,
                                     double score_tol) {
  PageRankValidation out;
  auto err = [&](const std::string& msg) {
    out.ok = false;
    out.errors.push_back(msg);
  };

  if (r.scores.size() != static_cast<std::size_t>(out_g.num_vertices)) {
    err("score size mismatch with graph");
    return out;
  }
  if (r.iterations < 0 || r.iterations > params.max_iters) {
    err("iterations out of range");
  }
  if (!std::isfinite(r.residual_l1) || r.residual_l1 < 0.0) {
    err("residual_l1 is not a finite non-negative value");
  }

  double sum = 0.0;
  for (double s : r.scores) {
    if (!std::isfinite(s) || s < 0.0) {
      err("non-finite or negative score");
      return out;
    }
    sum += s;
  }
  if (out_g.num_vertices > 0 && std::fabs(sum - 1.0) > 1e-4) {
    err("scores do not sum to ~1 (sum=" + std::to_string(sum) + ")");
  }
  if (std::fabs(sum - r.score_sum) > 1e-9) {
    err("score_sum field does not match sum of scores");
  }

  if (ref != nullptr) {
    if (ref->scores.size() != r.scores.size()) {
      err("reference score size mismatch");
      return out;
    }
    double l1 = 0.0;
    double linf = 0.0;
    for (std::size_t i = 0; i < r.scores.size(); ++i) {
      const double diff = std::fabs(r.scores[i] - ref->scores[i]);
      l1 += diff;
      if (diff > linf) {
        linf = diff;
      }
    }
    // L1 scales with V; compare mean absolute error as well as L∞.
    const double inv_n =
        1.0 / static_cast<double>(std::max<std::size_t>(r.scores.size(), 1));
    const double mae = l1 * inv_n;
    if (mae > score_tol) {
      err("mean |score-ref| exceeds tolerance (" + std::to_string(mae) +
          " > " + std::to_string(score_tol) + ")");
    }
    if (linf > score_tol) {
      err("Linf vs reference exceeds tolerance (" + std::to_string(linf) +
          " > " + std::to_string(score_tol) + ")");
    }
  }

  (void)params;
  return out;
}

}  // namespace pargraph
