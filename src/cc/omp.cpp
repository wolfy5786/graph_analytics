#include "kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

struct Cand {
  vtx_t v = 0;
  vtx_t lab = 0;
};

template <typename T>
void append_vec(std::vector<T>& dst, std::vector<T>& src) {
  if (src.empty()) {
    return;
  }
  dst.reserve(dst.size() + src.size());
  dst.insert(dst.end(), src.begin(), src.end());
  src.clear();
}

// Parallel binary reduction tree (same pattern as bfs_omp).
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

// Keep the minimum proposed label per vertex (deterministic).
void reduce_min_by_vertex(std::vector<Cand>& cands) {
  if (cands.empty()) {
    return;
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    if (a.v != b.v) {
      return a.v < b.v;
    }
    return a.lab < b.lab;
  });
  std::size_t w = 0;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    if (w == 0 || cands[i].v != cands[w - 1].v) {
      cands[w++] = cands[i];
    }
  }
  cands.resize(w);
}

}  // namespace

// OpenMP CC without atomics (BFS-omp style).
// Each round: threads scan owned out-arcs and emit undirected hook proposals
// (u gets label[v], v gets label[u]) into thread-local buffers; a reduction
// tree merges buffers; per-vertex min becomes the next label. Matches serial
// WCC on directed CSR because every stored arc contributes both ways.
CcResult cc_omp(const Graph& g, int num_threads) {
  CcResult r;
  const offset_t n = g.num_vertices;
  r.label.assign(static_cast<std::size_t>(n), 0);
  if (n == 0) {
    return r;
  }

  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  for (offset_t v = 0; v < n; ++v) {
    r.label[static_cast<std::size_t>(v)] = static_cast<vtx_t>(v);
  }

  std::uint64_t edges = 0;
  std::uint64_t rounds = 0;
  bool changed = true;

  while (changed) {
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Cand>> locals(static_cast<std::size_t>(nthreads));
    std::uint64_t edges_this = 0;

#pragma omp parallel reduction(+ : edges_this)
    {
      const int tid = omp_get_thread_num();
      auto& buf = locals[static_cast<std::size_t>(tid)];

#pragma omp for schedule(dynamic, 64)
      for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
        const vtx_t lu = r.label[static_cast<std::size_t>(u)];
        buf.push_back(Cand{static_cast<vtx_t>(u), lu});

        const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
        const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
        edges_this += static_cast<std::uint64_t>(end - begin);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          const vtx_t lv = r.label[static_cast<std::size_t>(v)];
          // Undirected hook proposals for WCC.
          buf.push_back(Cand{static_cast<vtx_t>(u), lv});
          buf.push_back(Cand{v, lu});
        }
      }
    }

    edges += edges_this;
    reduce_merge(locals);
    std::vector<Cand> cands = std::move(locals[0]);
    locals.clear();
    reduce_min_by_vertex(cands);

    int any = 0;
#pragma omp parallel for schedule(static) reduction(| : any)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(cands.size());
         ++i) {
      const Cand& c = cands[static_cast<std::size_t>(i)];
      if (c.lab < r.label[static_cast<std::size_t>(c.v)]) {
        r.label[static_cast<std::size_t>(c.v)] = c.lab;
        any = 1;
      }
    }

    ++rounds;
    changed = (any != 0);
  }

  // Canonicalize roots (component id = min vertex in the component).
#pragma omp parallel for schedule(static)
  for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
    vtx_t x = r.label[static_cast<std::size_t>(v)];
    while (r.label[static_cast<std::size_t>(x)] != x) {
      x = r.label[static_cast<std::size_t>(x)];
    }
    r.label[static_cast<std::size_t>(v)] = x;
  }

  r.edges_united = edges;
  r.rounds = rounds;
  cc_fill_aggregates(r);
  return r;
}

}  // namespace pargraph
