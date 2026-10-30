#include "kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace pargraph {
namespace {

void init_atomic_parents(std::vector<std::atomic<vtx_t>>& parent, offset_t n) {
  parent = std::vector<std::atomic<vtx_t>>(static_cast<std::size_t>(n));
  for (offset_t v = 0; v < n; ++v) {
    parent[static_cast<std::size_t>(v)].store(static_cast<vtx_t>(v),
                                              std::memory_order_relaxed);
  }
}

// atomicMin on a vtx_t cell. Returns true if the stored value decreased.
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

vtx_t find_compress(std::vector<std::atomic<vtx_t>>& parent, vtx_t x) {
  vtx_t r = x;
  vtx_t p = parent[static_cast<std::size_t>(r)].load(std::memory_order_relaxed);
  while (p != r) {
    r = p;
    p = parent[static_cast<std::size_t>(r)].load(std::memory_order_relaxed);
  }
  // Path compression (best-effort; concurrent finders may race harmlessly).
  vtx_t y = x;
  while (y != r) {
    vtx_t next =
        parent[static_cast<std::size_t>(y)].load(std::memory_order_relaxed);
    parent[static_cast<std::size_t>(y)].compare_exchange_weak(
        next, r, std::memory_order_relaxed, std::memory_order_relaxed);
    y = next;
  }
  return r;
}

void pointer_jump_round(std::vector<std::atomic<vtx_t>>& parent, offset_t n) {
#pragma omp parallel for schedule(static)
  for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
    const vtx_t p =
        parent[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
    const vtx_t gp =
        parent[static_cast<std::size_t>(p)].load(std::memory_order_relaxed);
    if (gp != p) {
      parent[static_cast<std::size_t>(v)].store(gp, std::memory_order_relaxed);
    }
  }
}

void snapshot_labels(const std::vector<std::atomic<vtx_t>>& parent,
                     std::vector<vtx_t>& label, offset_t n) {
  label.resize(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static)
  for (std::int64_t v = 0; v < static_cast<std::int64_t>(n); ++v) {
    vtx_t x =
        parent[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
    vtx_t p =
        parent[static_cast<std::size_t>(x)].load(std::memory_order_relaxed);
    while (p != x) {
      x = p;
      p = parent[static_cast<std::size_t>(x)].load(std::memory_order_relaxed);
    }
    label[static_cast<std::size_t>(v)] = x;
  }
}

CcResult labelprop_atomic(const Graph& g, int num_threads) {
  CcResult r;
  const offset_t n = g.num_vertices;
  if (n == 0) {
    return r;
  }
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  std::vector<std::atomic<vtx_t>> label;
  init_atomic_parents(label, n);

  std::uint64_t edges = 0;
  std::uint64_t rounds = 0;
  bool changed = true;

  while (changed) {
    changed = false;
    std::uint64_t edges_this = 0;
    int any = 0;

#pragma omp parallel reduction(+ : edges_this) reduction(| : any)
    {
#pragma omp for schedule(dynamic, 64)
      for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
        const vtx_t lu =
            label[static_cast<std::size_t>(u)].load(std::memory_order_relaxed);
        const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
        const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
        edges_this += static_cast<std::uint64_t>(end - begin);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          const vtx_t lv =
              label[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
          const vtx_t m = lu < lv ? lu : lv;
          if (atomic_min(label[static_cast<std::size_t>(u)], m)) {
            any = 1;
          }
          if (atomic_min(label[static_cast<std::size_t>(v)], m)) {
            any = 1;
          }
        }
      }
    }

    edges += edges_this;
    ++rounds;
    changed = (any != 0);

    // Shortcut chains so the next round sees shorter paths.
    pointer_jump_round(label, n);
  }

  snapshot_labels(label, r.label, n);
  r.edges_united = edges;
  r.rounds = rounds;
  cc_fill_aggregates(r);
  return r;
}

CcResult shiloach_vishkin(const Graph& g, int num_threads) {
  CcResult r;
  const offset_t n = g.num_vertices;
  if (n == 0) {
    return r;
  }
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  std::vector<std::atomic<vtx_t>> parent;
  init_atomic_parents(parent, n);

  std::uint64_t edges = 0;
  std::uint64_t rounds = 0;
  bool changed = true;

  while (changed) {
    changed = false;
    std::uint64_t edges_this = 0;
    int any = 0;

    // Hooking: for each arc, hook the larger root onto the smaller via CAS.
#pragma omp parallel reduction(+ : edges_this) reduction(| : any)
    {
#pragma omp for schedule(dynamic, 64)
      for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
        const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
        const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
        edges_this += static_cast<std::uint64_t>(end - begin);
        for (const auto* it = begin; it != end; ++it) {
          const vtx_t v = *it;
          vtx_t ru = find_compress(parent, static_cast<vtx_t>(u));
          vtx_t rv = find_compress(parent, v);
          while (ru != rv) {
            if (ru < rv) {
              vtx_t expected = rv;
              if (parent[static_cast<std::size_t>(rv)].compare_exchange_weak(
                      expected, ru, std::memory_order_relaxed,
                      std::memory_order_relaxed)) {
                any = 1;
                break;
              }
              rv = find_compress(parent, v);
              ru = find_compress(parent, static_cast<vtx_t>(u));
            } else {
              vtx_t expected = ru;
              if (parent[static_cast<std::size_t>(ru)].compare_exchange_weak(
                      expected, rv, std::memory_order_relaxed,
                      std::memory_order_relaxed)) {
                any = 1;
                break;
              }
              ru = find_compress(parent, static_cast<vtx_t>(u));
              rv = find_compress(parent, v);
            }
          }
        }
      }
    }

    edges += edges_this;
    ++rounds;
    pointer_jump_round(parent, n);
    changed = (any != 0);
  }

  snapshot_labels(parent, r.label, n);
  r.edges_united = edges;
  r.rounds = rounds;
  cc_fill_aggregates(r);
  return r;
}

// Afforest-style: sample the first k neighbors per vertex, compress, then
// finish only vertices whose component is still "small".
CcResult afforest(const Graph& g, int num_threads, int neighbor_samples = 2) {
  CcResult r;
  const offset_t n = g.num_vertices;
  if (n == 0) {
    return r;
  }
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
  if (neighbor_samples < 1) {
    neighbor_samples = 1;
  }

  std::vector<std::atomic<vtx_t>> parent;
  init_atomic_parents(parent, n);

  std::uint64_t edges = 0;
  std::uint64_t rounds = 0;

  auto hook_edge = [&](vtx_t u, vtx_t v) {
    vtx_t ru = find_compress(parent, u);
    vtx_t rv = find_compress(parent, v);
    while (ru != rv) {
      if (ru < rv) {
        vtx_t expected = rv;
        if (parent[static_cast<std::size_t>(rv)].compare_exchange_weak(
                expected, ru, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          return;
        }
        rv = find_compress(parent, v);
        ru = find_compress(parent, u);
      } else {
        vtx_t expected = ru;
        if (parent[static_cast<std::size_t>(ru)].compare_exchange_weak(
                expected, rv, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          return;
        }
        ru = find_compress(parent, u);
        rv = find_compress(parent, v);
      }
    }
  };

  // Phase 1: sample.
  {
    std::uint64_t edges_this = 0;
#pragma omp parallel reduction(+ : edges_this)
    {
#pragma omp for schedule(dynamic, 64)
      for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
        const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
        const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
        int taken = 0;
        for (const auto* it = begin; it != end && taken < neighbor_samples;
             ++it, ++taken) {
          hook_edge(static_cast<vtx_t>(u), *it);
          ++edges_this;
        }
      }
    }
    edges += edges_this;
    ++rounds;
    pointer_jump_round(parent, n);
    pointer_jump_round(parent, n);
  }

  // Phase 2: finish every remaining neighbor. Sampling first still shortens
  // parent chains (Afforest insight). Skipping large components after the
  // sample (classic Afforest) can miss bridges between two large components on
  // some graphs, so we keep the finish pass exact vs. serial.
  {
    std::uint64_t edges_this = 0;
#pragma omp parallel reduction(+ : edges_this)
    {
#pragma omp for schedule(dynamic, 64)
      for (std::int64_t u = 0; u < static_cast<std::int64_t>(n); ++u) {
        const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
        const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
        int skipped = 0;
        for (const auto* it = begin; it != end; ++it) {
          if (skipped < neighbor_samples) {
            ++skipped;
            continue;  // already hooked in the sample phase
          }
          hook_edge(static_cast<vtx_t>(u), *it);
          ++edges_this;
        }
      }
    }
    edges += edges_this;
    ++rounds;
  }

  // Final multi-jump compress.
  for (int i = 0; i < 4; ++i) {
    pointer_jump_round(parent, n);
  }
  snapshot_labels(parent, r.label, n);
  r.edges_united = edges;
  r.rounds = rounds;
  cc_fill_aggregates(r);
  return r;
}

}  // namespace

CcResult cc_omp_atomic(const Graph& g, int num_threads, const std::string& algo) {
  if (algo.empty() || algo == "labelprop") {
    return labelprop_atomic(g, num_threads);
  }
  if (algo == "sv") {
    return shiloach_vishkin(g, num_threads);
  }
  if (algo == "afforest") {
    return afforest(g, num_threads);
  }
  throw std::invalid_argument(
      "cc_omp_atomic: unknown algo '" + algo +
      "' (expected labelprop|sv|afforest)");
}

}  // namespace pargraph
