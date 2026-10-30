#include "bitmap.hpp"
#include "kernels.hpp"
#include "timer.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

void append_vtx(std::vector<vtx_t>& dst, std::vector<vtx_t>& src) {
  if (src.empty()) {
    return;
  }
  dst.reserve(dst.size() + src.size());
  dst.insert(dst.end(), src.begin(), src.end());
  src.clear();
}

void reduce_merge(std::vector<std::vector<vtx_t>>& locals) {
  const int n = static_cast<int>(locals.size());
  if (n <= 1) {
    return;
  }
  if (n <= 2) {
    for (int i = 1; i < n; ++i) {
      append_vtx(locals[0], locals[static_cast<std::size_t>(i)]);
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
          append_vtx(locals[static_cast<std::size_t>(i)],
                     locals[static_cast<std::size_t>(j)]);
        }
      }
    }
  }
}

bool try_visit(std::vector<std::atomic<std::uint8_t>>& visited, vtx_t v) {
  auto& cell = visited[static_cast<std::size_t>(v)];
  if (cell.load(std::memory_order_relaxed) != 0) {
    return false;
  }
  std::uint8_t expected = 0;
  return cell.compare_exchange_strong(expected, 1, std::memory_order_relaxed,
                                      std::memory_order_relaxed);
}

Graph build_reverse_csr(const Graph& g) {
  Graph rev;
  rev.num_vertices = g.num_vertices;
  rev.num_edges = g.num_edges;
  rev.directed = g.directed;
  rev.sorted = g.sorted;
  rev.offsets.assign(static_cast<std::size_t>(g.num_vertices) + 1, 0);

  for (offset_t u = 0; u < g.num_vertices; ++u) {
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      ++rev.offsets[static_cast<std::size_t>(*it) + 1];
    }
  }
  for (std::size_t i = 1; i < rev.offsets.size(); ++i) {
    rev.offsets[i] += rev.offsets[i - 1];
  }
  rev.neighbors.resize(static_cast<std::size_t>(g.num_edges));
  std::vector<offset_t> cursor(
      rev.offsets.begin(),
      rev.offsets.begin() + static_cast<std::ptrdiff_t>(g.num_vertices));
  for (offset_t u = 0; u < g.num_vertices; ++u) {
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      const vtx_t v = *it;
      const offset_t slot = cursor[static_cast<std::size_t>(v)]++;
      rev.neighbors[static_cast<std::size_t>(slot)] = static_cast<vtx_t>(u);
    }
  }
  if (rev.sorted) {
#pragma omp parallel for schedule(dynamic, 64)
    for (std::int64_t v = 0; v < static_cast<std::int64_t>(g.num_vertices);
         ++v) {
      auto* begin =
          rev.neighbors.data() + rev.offsets[static_cast<std::size_t>(v)];
      auto* end =
          rev.neighbors.data() + rev.offsets[static_cast<std::size_t>(v) + 1];
      std::sort(begin, end);
    }
  }
  return rev;
}

std::uint64_t frontier_scout(const Graph& g, const std::vector<vtx_t>& frontier) {
  std::uint64_t scout = 0;
  for (vtx_t v : frontier) {
    scout += g.degree(v);
  }
  return scout;
}

void rebuild_bitmap(Bitmap& bits, const std::vector<vtx_t>& frontier) {
  bits.clear();
  for (vtx_t v : frontier) {
    bits.set(v);
  }
}

enum class Dir { kTopDown, kBottomUp };

}  // namespace

BfsResult bfs_omp_diropt(const Graph& g, vtx_t source, int num_threads,
                         int alpha, int beta, bool per_level) {
  BfsResult r;
  r.dist.assign(static_cast<std::size_t>(g.num_vertices), kBfsUnreached);
  r.parent.assign(static_cast<std::size_t>(g.num_vertices), -1);

  if (g.num_vertices == 0 ||
      static_cast<offset_t>(source) >= g.num_vertices) {
    return r;
  }
  if (alpha < 1) {
    alpha = 15;
  }
  if (beta < 1) {
    beta = 18;
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  const Graph* in_graph = &g;
  Graph reverse;
  if (g.directed) {
    reverse = build_reverse_csr(g);
    in_graph = &reverse;
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
  Bitmap frontier_bits(static_cast<std::size_t>(g.num_vertices));
  frontier_bits.set(source);

  // GAPBS heuristic state.
  std::uint64_t edges_to_check = g.num_edges;
  std::uint64_t scout_count = g.degree(source);
  Dir dir = Dir::kTopDown;
  std::int32_t level = 0;

  while (!frontier.empty() && level < kBfsUnreached - 1) {
    // Decide direction for this level (GAPBS growing / shrinking rules).
    // Guard edges_to_check >= alpha so tiny graphs (E < alpha) stay top-down
    // under default parameters; integer truncation would otherwise make
    // edges/alpha == 0 and force bottom-up on every non-empty frontier.
    if (dir == Dir::kTopDown) {
      if (edges_to_check >= static_cast<std::uint64_t>(alpha) &&
          scout_count > edges_to_check / static_cast<std::uint64_t>(alpha)) {
        dir = Dir::kBottomUp;
      }
    } else if (frontier.size() < static_cast<std::size_t>(g.num_vertices) /
                                     static_cast<std::size_t>(beta)) {
      dir = Dir::kTopDown;
      scout_count = frontier_scout(g, frontier);
    }

    const std::int32_t next_dist = level + 1;
    const int nthreads = omp_get_max_threads();
    const bool bottom_up = (dir == Dir::kBottomUp);

    BfsLevelStat level_stat;
    level_stat.level = level;
    level_stat.frontier = frontier.size();
    level_stat.bottom_up = bottom_up;

    Timer level_timer;
    if (per_level) {
      level_timer.start();
    }

    std::uint64_t edges_this_level = 0;
    std::uint64_t newly_visited = 0;
    std::vector<vtx_t> next;

    if (!bottom_up) {
      std::vector<std::vector<vtx_t>> next_locals(
          static_cast<std::size_t>(nthreads));

#pragma omp parallel reduction(+ : edges_this_level, newly_visited)
      {
        const int tid = omp_get_thread_num();
        auto& local = next_locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(dynamic, 64)
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(frontier.size()); ++i) {
          const vtx_t u = frontier[static_cast<std::size_t>(i)];
          edges_this_level += g.degree(u);

          const auto* begin = g.neighbors_begin(u);
          const auto* end = g.neighbors_end(u);
          for (const auto* it = begin; it != end; ++it) {
            const vtx_t v = *it;
            if (!try_visit(visited, v)) {
              continue;
            }
            r.dist[v] = next_dist;
            r.parent[v] = static_cast<std::int32_t>(u);
            local.push_back(v);
            ++newly_visited;
          }
        }
      }

      r.edges_examined += edges_this_level;
      edges_to_check = (edges_to_check > scout_count)
                           ? (edges_to_check - scout_count)
                           : 0;
      r.visited += newly_visited;
      reduce_merge(next_locals);
      next = std::move(next_locals[0]);
      scout_count = frontier_scout(g, next);
    } else {
      // Bottom-up: each unvisited vertex looks for an in-neighbor on the
      // frontier bitmap. One iteration owns each v — no claim atomics.
      std::vector<std::vector<vtx_t>> next_locals(
          static_cast<std::size_t>(nthreads));

#pragma omp parallel reduction(+ : edges_this_level, newly_visited)
      {
        const int tid = omp_get_thread_num();
        auto& local = next_locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(dynamic, 256)
        for (std::int64_t vi = 0;
             vi < static_cast<std::int64_t>(g.num_vertices); ++vi) {
          const vtx_t v = static_cast<vtx_t>(vi);
          if (visited[static_cast<std::size_t>(v)].load(
                  std::memory_order_relaxed) != 0) {
            continue;
          }
          const auto* begin = in_graph->neighbors_begin(v);
          const auto* end = in_graph->neighbors_end(v);
          for (const auto* it = begin; it != end; ++it) {
            ++edges_this_level;
            const vtx_t u = *it;
            if (!frontier_bits.get(u)) {
              continue;
            }
            visited[static_cast<std::size_t>(v)].store(
                1, std::memory_order_relaxed);
            r.dist[static_cast<std::size_t>(v)] = next_dist;
            r.parent[static_cast<std::size_t>(v)] =
                static_cast<std::int32_t>(u);
            local.push_back(v);
            ++newly_visited;
            break;
          }
        }
      }

      r.edges_examined += edges_this_level;
      r.visited += newly_visited;
      reduce_merge(next_locals);
      next = std::move(next_locals[0]);
      scout_count = frontier_scout(g, next);
    }

    rebuild_bitmap(frontier_bits, next);

    if (per_level) {
      level_timer.stop();
      level_stat.edges_examined = edges_this_level;
      level_stat.time_ms = level_timer.elapsed_ms();
      r.levels.push_back(level_stat);
    }

    frontier = std::move(next);
    ++level;
  }

  return r;
}

}  // namespace pargraph
