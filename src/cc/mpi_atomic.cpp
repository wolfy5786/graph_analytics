#include "kernels.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
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

bool atomic_min(std::atomic<vtx_t>& cell, vtx_t cand) {
  vtx_t cur = cell.load(std::memory_order_relaxed);
  while (cand < cur) {
    if (cell.compare_exchange_weak(cur, cand, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

}  // namespace

// Hybrid MPI + OpenMP label propagation on a 1D vertex partition.
// Each rank hooks along edges of its owned vertices with OpenMP atomicMin,
// then Allreduce(MPI_MIN) merges labels across ranks. Matches serial WCC.
CcResult cc_mpi_atomic(const MpiVertexPart& part, int num_threads) {
  CcResult r;
  const offset_t n = part.num_vertices;
  r.label.assign(static_cast<std::size_t>(n), 0);
  if (n == 0) {
    return r;
  }

  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("MPI_Init required before cc_mpi_atomic");
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  std::vector<std::atomic<vtx_t>> label(static_cast<std::size_t>(n));
  for (offset_t v = 0; v < n; ++v) {
    label[static_cast<std::size_t>(v)].store(static_cast<vtx_t>(v),
                                             std::memory_order_relaxed);
  }

  // Scratch buffers for Allreduce (MPI needs contiguous non-atomic memory).
  std::vector<vtx_t> send_buf(static_cast<std::size_t>(n));
  std::vector<vtx_t> recv_buf(static_cast<std::size_t>(n));

  std::uint64_t edges = 0;
  std::uint64_t rounds = 0;
  double compute_sec = 0.0;
  double comm_sec = 0.0;
  std::uint64_t bytes_sent = 0;
  bool changed = true;

  const vtx_t lo = part.lo;
  const vtx_t hi = part.hi;
  const offset_t nlocal = static_cast<offset_t>(hi - lo);

  while (changed) {
    int local_any = 0;
    std::uint64_t edges_this = 0;

    const auto t_compute0 = Clock::now();
#pragma omp parallel reduction(+ : edges_this) reduction(| : local_any)
    {
#pragma omp for schedule(dynamic, 64)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(nlocal); ++i) {
        const vtx_t u = lo + static_cast<vtx_t>(i);
        const vtx_t lu =
            label[static_cast<std::size_t>(u)].load(std::memory_order_relaxed);
        // Local CSR stores only owned vertices; neighbor ids are global.
        const auto* begin = part.local.neighbors_begin(static_cast<vtx_t>(i));
        const auto* end = part.local.neighbors_end(static_cast<vtx_t>(i));
        edges_this += static_cast<std::uint64_t>(end - begin);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          const vtx_t lv =
              label[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
          const vtx_t m = lu < lv ? lu : lv;
          if (atomic_min(label[static_cast<std::size_t>(u)], m)) {
            local_any = 1;
          }
          if (atomic_min(label[static_cast<std::size_t>(v)], m)) {
            local_any = 1;
          }
        }
      }
    }
    compute_sec += seconds_since(t_compute0);
    edges += edges_this;

    // Pack labels for the global min-reduction.
#pragma omp parallel for schedule(static)
    for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
      send_buf[static_cast<std::size_t>(v)] =
          label[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
    }

    const auto t_comm0 = Clock::now();
    mpi_check(MPI_Allreduce(send_buf.data(), recv_buf.data(),
                            static_cast<int>(n), MPI_UINT32_T, MPI_MIN,
                            MPI_COMM_WORLD),
              "MPI_Allreduce labels");
    int global_any = 0;
    mpi_check(MPI_Allreduce(&local_any, &global_any, 1, MPI_INT, MPI_MAX,
                            MPI_COMM_WORLD),
              "MPI_Allreduce changed");
    comm_sec += seconds_since(t_comm0);
    bytes_sent += static_cast<std::uint64_t>(n) * sizeof(vtx_t);

#pragma omp parallel for schedule(static)
    for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
      label[static_cast<std::size_t>(v)].store(
          recv_buf[static_cast<std::size_t>(v)], std::memory_order_relaxed);
    }

    ++rounds;
    changed = (global_any != 0);
  }

  // Final compress to canonical roots.
#pragma omp parallel for schedule(static)
  for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
    vtx_t x = label[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
    vtx_t p = label[static_cast<std::size_t>(x)].load(std::memory_order_relaxed);
    while (p != x) {
      x = p;
      p = label[static_cast<std::size_t>(x)].load(std::memory_order_relaxed);
    }
    r.label[static_cast<std::size_t>(v)] = x;
  }

  // Slowest-rank phase split (matches BFS mpi_atomic reporting).
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

  r.edges_united = edges;
  r.rounds = rounds;
  r.mpi.compute_sec = compute_max;
  r.mpi.comm_sec = comm_max;
  r.mpi.bytes_sent = bytes_sum;
  cc_fill_aggregates(r);
  return r;
}

}  // namespace pargraph
