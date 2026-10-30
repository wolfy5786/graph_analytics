#include "kernels.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

void append_vtx(std::vector<vtx_t>& dst, std::vector<vtx_t>& src) {
  if (src.empty()) {
    return;
  }
  dst.insert(dst.end(), src.begin(), src.end());
  src.clear();
  src.shrink_to_fit();
}

// Parallel binary reduction tree: fold locals[1..] into locals[0].
void reduce_merge(std::vector<std::vector<vtx_t>>& locals) {
  const int n = static_cast<int>(locals.size());
  for (int stride = 1; stride < n; stride *= 2) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += 2 * stride) {
      const int j = i + stride;
      if (j < n) {
        append_vtx(locals[static_cast<std::size_t>(i)],
                   locals[static_cast<std::size_t>(j)]);
      }
    }
  }
}

// Atomic test-and-set: returns true iff this thread is first to mark v visited.
// Plain load before exchange cuts contention on already-reached neighbors.
bool try_visit(std::vector<std::atomic<std::uint8_t>>& visited, vtx_t v) {
  auto& cell = visited[static_cast<std::size_t>(v)];
  if (cell.load(std::memory_order_relaxed) != 0) {
    return false;
  }
  std::uint8_t expected = 0;
  return cell.compare_exchange_strong(expected, 1, std::memory_order_relaxed,
                                      std::memory_order_relaxed);
}

}  // namespace

BfsResult bfs_omp_atomic(const Graph& g, vtx_t source, int num_threads) {
  BfsResult r;
  r.dist.assign(static_cast<std::size_t>(g.num_vertices), kBfsUnreached);
  r.parent.assign(static_cast<std::size_t>(g.num_vertices), -1);

  if (static_cast<offset_t>(source) >= g.num_vertices) {
    return r;
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  std::vector<std::atomic<std::uint8_t>> visited(
      static_cast<std::size_t>(g.num_vertices));
  for (auto& cell : visited) {
    cell.store(0, std::memory_order_relaxed);
  }

  visited[source].store(1, std::memory_order_relaxed);
  r.dist[source] = 0;
  r.parent[source] = static_cast<std::int32_t>(source);
  r.visited = 1;

  std::vector<vtx_t> frontier;
  frontier.push_back(source);
  std::int32_t level = 0;

  while (!frontier.empty()) {
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<vtx_t>> next_locals(
        static_cast<std::size_t>(nthreads));
    std::uint64_t edges_this_level = 0;
    std::uint64_t newly_visited = 0;
    const std::int32_t next_dist = level + 1;

    // Process current level in parallel. Winning TAS thread alone writes
    // dist/parent and appends to its thread-local next frontier.
#pragma omp parallel reduction(+ : edges_this_level, newly_visited)
    {
      const int tid = omp_get_thread_num();
      auto& next = next_locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(dynamic, 64)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(frontier.size());
           ++i) {
        const vtx_t u = frontier[static_cast<std::size_t>(i)];
        edges_this_level += g.degree(u);

        const auto* begin = g.neighbors_begin(u);
        const auto* end = g.neighbors_end(u);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          if (!try_visit(visited, v)) {
            continue;
          }
          // Sole discoverer: no races on dist/parent for v.
          r.dist[v] = next_dist;
          r.parent[v] = static_cast<std::int32_t>(u);
          next.push_back(v);
          ++newly_visited;
        }
      }
    }

    r.edges_examined += edges_this_level;
    r.visited += newly_visited;

    // Barrier between levels: merge thread-local next vectors, then swap.
    reduce_merge(next_locals);
    frontier = std::move(next_locals[0]);
    ++level;
  }

  return r;
}

}  // namespace pargraph
