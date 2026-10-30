#include "kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

namespace pargraph {
namespace {

void mpi_check(int rc, const char* what) {
  if (rc == MPI_SUCCESS) {
    return;
  }
  char buf[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  MPI_Error_string(rc, buf, &len);
  throw std::runtime_error(std::string(what) + ": " +
                           std::string(buf, static_cast<std::size_t>(len)));
}

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Matches bfs/mpi_atomic.cpp range_lo.
offset_t range_lo(int r, offset_t n, int ranks) {
  const offset_t nr = static_cast<offset_t>(ranks);
  const offset_t base = n / nr;
  const offset_t rem = n % nr;
  const auto rr = static_cast<offset_t>(r);
  if (rr <= rem) {
    return rr * (base + 1);
  }
  return rem * (base + 1) + (rr - rem) * base;
}

}  // namespace

PageRankResult pagerank_mpi(const MpiVertexPart& part, const Graph& out_g,
                            const Graph& in_g, PageRankParams params,
                            int num_threads) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("MPI_Init required before pagerank_mpi");
  }

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
  std::vector<double> gathered(static_cast<std::size_t>(n), 0.0);

  // Allgatherv counts for owned score slices (same 1D ranges as BFS).
  std::vector<int> recvcounts(static_cast<std::size_t>(part.ranks), 0);
  std::vector<int> displs(static_cast<std::size_t>(part.ranks), 0);
  for (int rr = 0; rr < part.ranks; ++rr) {
    const offset_t rlo = range_lo(rr, n, part.ranks);
    const offset_t rhi = range_lo(rr + 1, n, part.ranks);
    recvcounts[static_cast<std::size_t>(rr)] = static_cast<int>(rhi - rlo);
    displs[static_cast<std::size_t>(rr)] = static_cast<int>(rlo);
  }

  const vtx_t lo = part.lo;
  const vtx_t hi = part.hi;
  const int sendcount = static_cast<int>(hi - lo);
  std::vector<double> local_next(static_cast<std::size_t>(
      sendcount > 0 ? sendcount : 0));
  double compute_sec = 0.0;
  double comm_sec = 0.0;
  std::uint64_t bytes_sent = 0;

  for (int it = 0; it < params.max_iters; ++it) {
    const auto t_compute0 = Clock::now();

    double dangling_local = 0.0;
#pragma omp parallel for reduction(+ : dangling_local) schedule(static)
    for (std::int64_t u = static_cast<std::int64_t>(lo);
         u < static_cast<std::int64_t>(hi); ++u) {
      if (out_g.degree(static_cast<vtx_t>(u)) == 0) {
        dangling_local += curr[static_cast<std::size_t>(u)];
      }
    }

    compute_sec += seconds_since(t_compute0);
    const auto t_comm0 = Clock::now();
    double dangling = 0.0;
    mpi_check(MPI_Allreduce(&dangling_local, &dangling, 1, MPI_DOUBLE, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce dangling");
    comm_sec += seconds_since(t_comm0);
    bytes_sent += sizeof(double);

    const double base = (1.0 - d) * inv_n + d * dangling * inv_n;
    const auto t_compute1 = Clock::now();
    double residual_local = 0.0;

#pragma omp parallel for reduction(+ : residual_local) schedule(dynamic, 64)
    for (std::int64_t v = static_cast<std::int64_t>(lo);
         v < static_cast<std::int64_t>(hi); ++v) {
      double sum = 0.0;
      const auto* begin = in_g.neighbors_begin(static_cast<vtx_t>(v));
      const auto* end = in_g.neighbors_end(static_cast<vtx_t>(v));
      for (const auto* itn = begin; itn != end; ++itn) {
        const vtx_t u = *itn;
        const offset_t deg = out_g.degree(u);
        sum += curr[static_cast<std::size_t>(u)] / static_cast<double>(deg);
      }
      const double nv = base + d * sum;
      residual_local += std::fabs(nv - curr[static_cast<std::size_t>(v)]);
      local_next[static_cast<std::size_t>(v - static_cast<std::int64_t>(lo))] =
          nv;
    }
    compute_sec += seconds_since(t_compute1);

    const auto t_comm1 = Clock::now();
    mpi_check(MPI_Allgatherv(local_next.data(), sendcount, MPI_DOUBLE,
                             gathered.data(), recvcounts.data(), displs.data(),
                             MPI_DOUBLE, MPI_COMM_WORLD),
              "MPI_Allgatherv scores");
    double residual = 0.0;
    mpi_check(MPI_Allreduce(&residual_local, &residual, 1, MPI_DOUBLE, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce residual");
    comm_sec += seconds_since(t_comm1);
    bytes_sent += static_cast<std::uint64_t>(sendcount) * sizeof(double);
    bytes_sent += sizeof(double);

    curr.swap(gathered);
    r.iterations = it + 1;
    r.residual_l1 = residual;
    r.edges_examined += out_g.num_edges;
    if (residual < params.tolerance) {
      break;
    }
  }

  // Max phase times across ranks (BFS-style slowest-rank reporting).
  double compute_max = 0.0;
  double comm_max = 0.0;
  std::uint64_t bytes_sum = 0;
  mpi_check(MPI_Allreduce(&compute_sec, &compute_max, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce compute");
  mpi_check(MPI_Allreduce(&comm_sec, &comm_max, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce comm");
  mpi_check(MPI_Allreduce(&bytes_sent, &bytes_sum, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce bytes");
  r.mpi.compute_sec = compute_max;
  r.mpi.comm_sec = comm_max;
  r.mpi.bytes_sent = bytes_sum;

  r.scores = std::move(curr);
  double sum = 0.0;
  for (double s : r.scores) {
    sum += s;
  }
  r.score_sum = sum;
  return r;
}

}  // namespace pargraph
