#include "kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
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

// MPI int counts: discovery traffic is at most 2 * V uint32s per rank.
constexpr offset_t kMaxMpiVertices = 1000000000ull;

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

// First `rem` ranks own (base+1) vertices; the rest own `base`.
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

// O(1) owner. Matches range_lo for every v in [0, n).
int owner_of(vtx_t v, vtx_t base, vtx_t rem) {
  if (base == 0) {
    return static_cast<int>(v);
  }
  const offset_t span = static_cast<offset_t>(base) + 1;
  const offset_t big = span * static_cast<offset_t>(rem);
  if (static_cast<offset_t>(v) < big) {
    return static_cast<int>(static_cast<offset_t>(v) / span);
  }
  return static_cast<int>(rem) +
         static_cast<int>((static_cast<offset_t>(v) - big) / base);
}

template <typename T>
void append_vec(std::vector<T>& dst, std::vector<T>& src) {
  if (src.empty()) {
    return;
  }
  dst.reserve(dst.size() + src.size());
  dst.insert(dst.end(), src.begin(), src.end());
  src.clear();
}

template <typename T>
void reduce_merge(std::vector<std::vector<T>>& locals) {
  const int n = static_cast<int>(locals.size());
  if (n <= 1) {
    return;
  }
  if (n <= 2) {
    for (int i = 1; i < n; ++i) {
      append_vec(locals[0], locals[static_cast<std::size_t>(i)]);
    }
    return;
  }
#pragma omp parallel
  {
    for (int stride = 1; stride < n; stride *= 2) {
      const int step = 2 * stride;
      const int nmerges = (n + step - 1) / step;
#pragma omp for schedule(static)
      for (int k = 0; k < nmerges; ++k) {
        const int i = k * step;
        const int j = i + stride;
        if (j < n) {
          append_vec(locals[static_cast<std::size_t>(i)],
                     locals[static_cast<std::size_t>(j)]);
        }
      }
    }
  }
}

bool try_visit(std::vector<std::atomic<std::uint8_t>>& seen, vtx_t v) {
  auto& cell = seen[static_cast<std::size_t>(v)];
  if (cell.load(std::memory_order_relaxed) != 0) {
    return false;
  }
  std::uint8_t expected = 0;
  return cell.compare_exchange_strong(expected, 1, std::memory_order_relaxed,
                                      std::memory_order_relaxed);
}

std::vector<vtx_t> take_merged(std::vector<std::vector<vtx_t>>& locals) {
  reduce_merge(locals);
  std::vector<vtx_t> out = std::move(locals[0]);
  for (auto& bucket : locals) {
    bucket.clear();
  }
  return out;
}

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

}  // namespace

MpiVertexPart mpi_partition_vertices_1d(const Graph& g) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error(
        "MPI_Init required before mpi_partition_vertices_1d");
  }
  if (g.num_vertices > kMaxMpiVertices) {
    throw std::runtime_error(
        "mpi atomic BFS supports up to 1e9 vertices (MPI int counts)");
  }

  MpiVertexPart part;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &part.rank), "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &part.ranks), "MPI_Comm_size");
  if (part.ranks < 1) {
    throw std::runtime_error("MPI_Comm_size < 1");
  }
  part.num_vertices = g.num_vertices;
  part.local.directed = g.directed;
  part.local.sorted = g.sorted;

  if (g.num_vertices == 0) {
    part.local.offsets.assign(1, 0);
    return part;
  }

  const offset_t n = g.num_vertices;
  const offset_t nr = static_cast<offset_t>(part.ranks);
  part.base = static_cast<vtx_t>(n / nr);
  part.rem = static_cast<vtx_t>(n % nr);
  const offset_t lo = range_lo(part.rank, n, part.ranks);
  const offset_t hi = range_lo(part.rank + 1, n, part.ranks);
  part.lo = static_cast<vtx_t>(lo);
  part.hi = static_cast<vtx_t>(hi);

  const offset_t nlocal = hi - lo;
  part.local.num_vertices = nlocal;
  if (nlocal == 0) {
    part.local.offsets.assign(1, 0);
    return part;
  }

  const offset_t edge_lo = g.offsets[static_cast<std::size_t>(lo)];
  const offset_t edge_hi = g.offsets[static_cast<std::size_t>(hi)];
  part.local.num_edges = edge_hi - edge_lo;
  part.local.offsets.resize(static_cast<std::size_t>(nlocal) + 1);
  for (offset_t i = 0; i <= nlocal; ++i) {
    part.local.offsets[static_cast<std::size_t>(i)] =
        g.offsets[static_cast<std::size_t>(lo + i)] - edge_lo;
  }
  part.local.neighbors.assign(
      g.neighbors.begin() + static_cast<std::ptrdiff_t>(edge_lo),
      g.neighbors.begin() + static_cast<std::ptrdiff_t>(edge_hi));
  return part;
}

BfsResult bfs_mpi_atomic(const MpiVertexPart& part, vtx_t source,
                         int num_threads) {
  BfsResult r;
  const std::size_t n = static_cast<std::size_t>(part.num_vertices);
  r.dist.assign(n, kBfsUnreached);
  r.parent.assign(n, -1);

  if (part.num_vertices == 0 ||
      static_cast<offset_t>(source) >= part.num_vertices) {
    return r;
  }

  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("MPI_Init required before bfs_mpi_atomic");
  }

  int world_rank = 0;
  int world_ranks = 0;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &world_ranks), "MPI_Comm_size");
  int mismatch = (world_rank != part.rank || world_ranks != part.ranks) ? 1 : 0;
  int any_mismatch = 0;
  mpi_check(MPI_Allreduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce partition check");
  if (any_mismatch) {
    throw std::runtime_error("MpiVertexPart does not match MPI_COMM_WORLD");
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
  const int nthreads = std::max(1, omp_get_max_threads());

  double compute = 0.0;
  double comm = 0.0;
  std::uint64_t bytes = 0;
  auto t_phase = Clock::now();

  std::vector<std::atomic<std::uint8_t>> seen(n);
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    seen[static_cast<std::size_t>(i)].store(0, std::memory_order_relaxed);
  }

  std::vector<vtx_t> frontier;
  std::uint64_t local_visited = 0;
  std::uint64_t local_edges = 0;
  const int source_owner = owner_of(source, part.base, part.rem);
  if (part.rank == source_owner) {
    seen[source].store(1, std::memory_order_relaxed);
    r.dist[source] = 0;
    r.parent[source] = static_cast<std::int32_t>(source);
    frontier.push_back(source);
    local_visited = 1;
  }
  compute += seconds_since(t_phase);

  std::vector<std::vector<vtx_t>> next_locals(static_cast<std::size_t>(nthreads));
  std::vector<std::vector<std::vector<std::uint32_t>>> send_locals(
      static_cast<std::size_t>(nthreads),
      std::vector<std::vector<std::uint32_t>>(
          static_cast<std::size_t>(part.ranks)));

  std::int32_t level = 0;
  while (true) {
    t_phase = Clock::now();
    std::uint64_t edges_this_level = 0;
    std::uint64_t newly_visited = 0;
    const std::int32_t next_dist = level + 1;

#pragma omp parallel reduction(+ : edges_this_level, newly_visited)
    {
      const int tid = omp_get_thread_num();
      auto& next = next_locals[static_cast<std::size_t>(tid)];
      auto& sends = send_locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(dynamic, 64)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(frontier.size());
           ++i) {
        const vtx_t u = frontier[static_cast<std::size_t>(i)];
        const vtx_t u_local = static_cast<vtx_t>(u - part.lo);
        edges_this_level += part.local.degree(u_local);

        const auto* begin = part.local.neighbors_begin(u_local);
        const auto* end = part.local.neighbors_end(u_local);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          if (static_cast<offset_t>(v) >= part.num_vertices) {
            continue;
          }
          const int dest = owner_of(v, part.base, part.rem);
          if (dest == part.rank) {
            if (!try_visit(seen, v)) {
              continue;
            }
            r.dist[v] = next_dist;
            r.parent[v] = static_cast<std::int32_t>(u);
            next.push_back(v);
            ++newly_visited;
          } else if (dest >= 0 && dest < part.ranks && try_visit(seen, v)) {
            auto& buf = sends[static_cast<std::size_t>(dest)];
            buf.push_back(v);
            buf.push_back(u);
          }
        }
      }
    }

    local_edges += edges_this_level;
    local_visited += newly_visited;

    std::vector<vtx_t> next = take_merged(next_locals);

    std::vector<int> send_counts(static_cast<std::size_t>(part.ranks), 0);
    std::int64_t send_total = 0;
    for (int t = 0; t < nthreads; ++t) {
      for (int d = 0; d < part.ranks; ++d) {
        const auto len = send_locals[static_cast<std::size_t>(t)]
                                    [static_cast<std::size_t>(d)]
                                        .size();
        send_counts[static_cast<std::size_t>(d)] += static_cast<int>(len);
        send_total += static_cast<std::int64_t>(len);
      }
    }
    std::vector<std::uint32_t> sendbuf(static_cast<std::size_t>(send_total));
    std::vector<int> sdispls(static_cast<std::size_t>(part.ranks), 0);
    int cursor = 0;
    for (int d = 0; d < part.ranks; ++d) {
      sdispls[static_cast<std::size_t>(d)] = cursor;
      for (int t = 0; t < nthreads; ++t) {
        auto& buf = send_locals[static_cast<std::size_t>(t)]
                               [static_cast<std::size_t>(d)];
        if (buf.empty()) {
          continue;
        }
        std::copy(buf.begin(), buf.end(), sendbuf.begin() + cursor);
        cursor += static_cast<int>(buf.size());
        buf.clear();
      }
    }
    bytes += static_cast<std::uint64_t>(send_total) * sizeof(std::uint32_t);
    compute += seconds_since(t_phase);

    t_phase = Clock::now();
    std::vector<int> recv_counts(static_cast<std::size_t>(part.ranks), 0);
    std::uint32_t dummy_u = 0;
    const std::uint32_t* send_ptr = sendbuf.empty() ? &dummy_u : sendbuf.data();
    mpi_check(MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(),
                           1, MPI_INT, MPI_COMM_WORLD),
              "MPI_Alltoall");

    std::vector<int> rdispls(static_cast<std::size_t>(part.ranks), 0);
    int recv_total = 0;
    for (int d = 0; d < part.ranks; ++d) {
      rdispls[static_cast<std::size_t>(d)] = recv_total;
      recv_total += recv_counts[static_cast<std::size_t>(d)];
    }
    std::vector<std::uint32_t> recvbuf(static_cast<std::size_t>(
        recv_total > 0 ? recv_total : 0));
    std::uint32_t* recv_ptr = recvbuf.empty() ? &dummy_u : recvbuf.data();
    mpi_check(MPI_Alltoallv(send_ptr, send_counts.data(), sdispls.data(),
                            MPI_UINT32_T, recv_ptr, recv_counts.data(),
                            rdispls.data(), MPI_UINT32_T, MPI_COMM_WORLD),
              "MPI_Alltoallv");
    comm += seconds_since(t_phase);

    t_phase = Clock::now();
    const int npairs = recv_total / 2;
    std::uint64_t claimed = 0;
#pragma omp parallel reduction(+ : claimed)
    {
      const int tid = omp_get_thread_num();
      auto& bucket = next_locals[static_cast<std::size_t>(tid)];
#pragma omp for schedule(dynamic, 64)
      for (int i = 0; i < npairs; ++i) {
        const vtx_t v = recvbuf[static_cast<std::size_t>(2 * i)];
        const vtx_t parent_u = recvbuf[static_cast<std::size_t>(2 * i + 1)];
        if (v < part.lo || v >= part.hi) {
          continue;
        }
        if (!try_visit(seen, v)) {
          continue;
        }
        r.dist[v] = next_dist;
        r.parent[v] = static_cast<std::int32_t>(parent_u);
        bucket.push_back(v);
        ++claimed;
      }
    }
    local_visited += claimed;
    std::vector<vtx_t> remote_next = take_merged(next_locals);
    if (!remote_next.empty()) {
      next.insert(next.end(), remote_next.begin(), remote_next.end());
    }
    compute += seconds_since(t_phase);

    t_phase = Clock::now();
    std::uint64_t local_next = static_cast<std::uint64_t>(next.size());
    std::uint64_t global_next = 0;
    mpi_check(MPI_Allreduce(&local_next, &global_next, 1, MPI_UINT64_T, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce frontier");
    comm += seconds_since(t_phase);

    if (global_next == 0) {
      break;
    }
    frontier.swap(next);
    ++level;
  }

  t_phase = Clock::now();
  std::uint64_t vis_sum = 0;
  std::uint64_t edge_sum = 0;
  mpi_check(MPI_Allreduce(&local_visited, &vis_sum, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce visited");
  mpi_check(MPI_Allreduce(&local_edges, &edge_sum, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce edges");
  r.visited = vis_sum;
  r.edges_examined = edge_sum;

  std::vector<int> counts(static_cast<std::size_t>(part.ranks));
  std::vector<int> displs(static_cast<std::size_t>(part.ranks));
  int off = 0;
  for (int rr = 0; rr < part.ranks; ++rr) {
    const offset_t a = range_lo(rr, part.num_vertices, part.ranks);
    const offset_t b = range_lo(rr + 1, part.num_vertices, part.ranks);
    counts[static_cast<std::size_t>(rr)] = static_cast<int>(b - a);
    displs[static_cast<std::size_t>(rr)] = off;
    off += counts[static_cast<std::size_t>(rr)];
  }
  const int local_n = static_cast<int>(part.hi - part.lo);
  std::vector<std::int32_t> dist_slice(static_cast<std::size_t>(local_n));
  std::vector<std::int32_t> parent_slice(static_cast<std::size_t>(local_n));
  for (int i = 0; i < local_n; ++i) {
    dist_slice[static_cast<std::size_t>(i)] =
        r.dist[static_cast<std::size_t>(part.lo) + static_cast<std::size_t>(i)];
    parent_slice[static_cast<std::size_t>(i)] =
        r.parent[static_cast<std::size_t>(part.lo) +
                 static_cast<std::size_t>(i)];
  }
  std::int32_t dummy_i = 0;
  const std::int32_t* dist_send = local_n > 0 ? dist_slice.data() : &dummy_i;
  const std::int32_t* parent_send =
      local_n > 0 ? parent_slice.data() : &dummy_i;
  mpi_check(MPI_Allgatherv(dist_send, local_n, MPI_INT32_T, r.dist.data(),
                           counts.data(), displs.data(), MPI_INT32_T,
                           MPI_COMM_WORLD),
            "MPI_Allgatherv dist");
  mpi_check(MPI_Allgatherv(parent_send, local_n, MPI_INT32_T, r.parent.data(),
                           counts.data(), displs.data(), MPI_INT32_T,
                           MPI_COMM_WORLD),
            "MPI_Allgatherv parent");
  comm += seconds_since(t_phase);

  std::vector<double> all_compute(static_cast<std::size_t>(part.ranks));
  std::vector<double> all_comm(static_cast<std::size_t>(part.ranks));
  mpi_check(MPI_Allgather(&compute, 1, MPI_DOUBLE, all_compute.data(), 1,
                          MPI_DOUBLE, MPI_COMM_WORLD),
            "MPI_Allgather compute");
  mpi_check(MPI_Allgather(&comm, 1, MPI_DOUBLE, all_comm.data(), 1, MPI_DOUBLE,
                          MPI_COMM_WORLD),
            "MPI_Allgather comm");
  int slow = 0;
  double busy = -1.0;
  for (int i = 0; i < part.ranks; ++i) {
    const double b = all_compute[static_cast<std::size_t>(i)] +
                     all_comm[static_cast<std::size_t>(i)];
    if (b > busy) {
      busy = b;
      slow = i;
    }
  }
  r.mpi.compute_sec = all_compute[static_cast<std::size_t>(slow)];
  r.mpi.comm_sec = all_comm[static_cast<std::size_t>(slow)];

  std::uint64_t bytes_sum = 0;
  mpi_check(MPI_Allreduce(&bytes, &bytes_sum, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce bytes");
  r.mpi.bytes_sent = bytes_sum;
  return r;
}

}  // namespace pargraph
