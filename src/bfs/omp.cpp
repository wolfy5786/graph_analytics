#include "kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

// Candidate discovery record: neighbor + discovering parent from prior level.
struct Cand {
  vtx_t v = 0;
  vtx_t parent = 0;
};

// Append src onto dst and clear src. Capacity is retained (no shrink_to_fit)
// so merge rounds do not thrash the allocator.
template <typename T>
void append_vec(std::vector<T>& dst, std::vector<T>& src) {
  if (src.empty()) {
    return;
  }
  dst.reserve(dst.size() + src.size());
  dst.insert(dst.end(), src.begin(), src.end());
  src.clear();
}

// Parallel binary reduction tree: fold locals[1..] into locals[0].
// One OpenMP parallel region for the whole tree; each stride uses an omp for
// (implicit barrier) so merge rounds stay race-free without re-forking.
template <typename T>
void reduce_merge(std::vector<std::vector<T>>& locals) {
  const int n = static_cast<int>(locals.size());
  if (n <= 1) {
    return;
  }

  // Tiny teams: serial fold avoids fork/join overhead entirely.
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
      // Implicit barrier at end of omp for: next stride sees completed merges.
    }
  }
}

// Sort by (vertex, parent) ascending, then keep the first entry per vertex
// (= smallest parent). Guarantees a deterministic valid BFS parent among
// same-level discoverers.
void dedupe_by_vertex(std::vector<Cand>& cands) {
  if (cands.empty()) {
    return;
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    if (a.v != b.v) {
      return a.v < b.v;
    }
    return a.parent < b.parent;
  });
  std::size_t w = 0;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    if (w == 0 || cands[i].v != cands[w - 1].v) {
      cands[w++] = cands[i];
    }
  }
  cands.resize(w);
}

// parent[] / dist[] are int32_t. Vertex IDs and distances must fit that schema
// (dist uses kBfsUnreached == INT32_MAX as the sentinel).
bool vtx_fits_parent_field(vtx_t v) {
  return static_cast<offset_t>(v) <=
         static_cast<offset_t>(std::numeric_limits<std::int32_t>::max());
}

}  // namespace

BfsResult bfs_omp(const Graph& g, vtx_t source, int num_threads) {
  BfsResult r;
  r.dist.assign(static_cast<std::size_t>(g.num_vertices), kBfsUnreached);
  r.parent.assign(static_cast<std::size_t>(g.num_vertices), -1);

  // Bounds: compare in offset_t (unsigned 64-bit) so large vtx_t / empty
  // graphs (num_vertices == 0) are handled without signed wrap surprises.
  if (g.num_vertices == 0 ||
      static_cast<offset_t>(source) >= g.num_vertices) {
    return r;
  }
  // Result schema stores parent as int32_t; refuse IDs that cannot be stored.
  if (!vtx_fits_parent_field(source)) {
    return r;
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  r.dist[source] = 0;
  r.parent[source] = static_cast<std::int32_t>(source);
  r.visited = 1;

  std::vector<vtx_t> frontier;
  frontier.push_back(source);
  std::int32_t level = 0;

  while (!frontier.empty()) {
    // Next distance must stay strictly below kBfsUnreached (INT32_MAX).
    if (level >= kBfsUnreached - 1) {
      break;
    }
    const std::int32_t next_dist = level + 1;

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Cand>> locals(static_cast<std::size_t>(nthreads));
    std::uint64_t edges_this_level = 0;

    // Phase 1: neighbor discovery into thread-local buffers. Read dist[] only;
    // do not mark visited and do not use atomics on the visited/dist state.
#pragma omp parallel reduction(+ : edges_this_level)
    {
      const int tid = omp_get_thread_num();
      auto& buf = locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(dynamic, 64)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(frontier.size());
           ++i) {
        const vtx_t u = frontier[static_cast<std::size_t>(i)];
        edges_this_level += g.degree(u);

        const auto* begin = g.neighbors_begin(u);
        const auto* end = g.neighbors_end(u);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          // Non-atomic snapshot: duplicates across threads are OK.
          if (r.dist[static_cast<std::size_t>(v)] == kBfsUnreached) {
            buf.push_back(Cand{v, u});
          }
        }
      }
    }
    r.edges_examined += edges_this_level;

    // Phase 2: parallel reduction-tree merge of local candidate vectors.
    reduce_merge(locals);
    std::vector<Cand> cands = std::move(locals[0]);
    locals.clear();

    // Phase 3: deduplicate (deterministic: smallest parent per vertex).
    dedupe_by_vertex(cands);

    // Implicit sync: discovery + merge + dedupe finished before claiming.
    // Phase 4: check / mark unique vertices; build next frontier.
    // After dedupe each vertex appears once, so dist/parent writes do not race.
    std::vector<std::vector<vtx_t>> next_locals(
        static_cast<std::size_t>(nthreads));
    std::uint64_t newly_visited = 0;

#pragma omp parallel reduction(+ : newly_visited)
    {
      const int tid = omp_get_thread_num();
      auto& next_buf = next_locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(static)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(cands.size());
           ++i) {
        const Cand& c = cands[static_cast<std::size_t>(i)];
        if (r.dist[static_cast<std::size_t>(c.v)] != kBfsUnreached) {
          continue;
        }
        // Parent IDs beyond int32_t cannot be stored in BfsResult::parent.
        if (!vtx_fits_parent_field(c.parent) || !vtx_fits_parent_field(c.v)) {
          continue;
        }
        r.dist[static_cast<std::size_t>(c.v)] = next_dist;
        r.parent[static_cast<std::size_t>(c.v)] =
            static_cast<std::int32_t>(c.parent);
        next_buf.push_back(c.v);
        ++newly_visited;
      }
    }

    r.visited += newly_visited;

    reduce_merge(next_locals);
    frontier = std::move(next_locals[0]);

    ++level;
  }

  return r;
}

}  // namespace pargraph
