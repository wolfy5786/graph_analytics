#include "intersect.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace pargraph {
namespace {

bool better_endpoint(vtx_t u, vtx_t v, offset_t du, offset_t dv,
                     TcOrientMode orient) {
  if (orient == TcOrientMode::Id) {
    return u < v;
  }
  // Degree: lower degree → higher degree; break ties by id.
  if (du != dv) {
    return du < dv;
  }
  return u < v;
}

std::uint64_t count_on_oriented(const Graph& directed, IntersectMode mode,
                                std::uint64_t* wedges_out,
                                std::uint64_t* intersect_ops_out) {
  std::uint64_t triangles = 0;
  std::uint64_t wedges = 0;
  std::uint64_t ops = 0;
  for (offset_t u = 0; u < directed.num_vertices; ++u) {
    const auto* ub = directed.neighbors_begin(static_cast<vtx_t>(u));
    const auto* ue = directed.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = ub; it != ue; ++it) {
      const vtx_t v = *it;
      ++wedges;
      ++ops;
      triangles += intersect_sorted(
          ub, ue, directed.neighbors_begin(v), directed.neighbors_end(v),
          mode);
    }
  }
  if (wedges_out) {
    *wedges_out = wedges;
  }
  if (intersect_ops_out) {
    *intersect_ops_out = ops;
  }
  return triangles;
}

}  // namespace

Graph orient_graph(const Graph& g, TcOrientMode orient) {
  Graph out;
  out.num_vertices = g.num_vertices;
  out.directed = true;
  out.sorted = true;
  out.offsets.assign(static_cast<std::size_t>(g.num_vertices) + 1, 0);

  // Count forward out-degree.
  for (offset_t u = 0; u < g.num_vertices; ++u) {
    const offset_t du = g.degree(static_cast<vtx_t>(u));
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      const vtx_t v = *it;
      if (better_endpoint(static_cast<vtx_t>(u), v, du, g.degree(v), orient)) {
        ++out.offsets[static_cast<std::size_t>(u) + 1];
      }
    }
  }
  for (offset_t v = 0; v < g.num_vertices; ++v) {
    out.offsets[static_cast<std::size_t>(v) + 1] +=
        out.offsets[static_cast<std::size_t>(v)];
  }
  out.num_edges = out.offsets.back();
  out.neighbors.resize(static_cast<std::size_t>(out.num_edges));

  std::vector<offset_t> cursor = out.offsets;
  for (offset_t u = 0; u < g.num_vertices; ++u) {
    const offset_t du = g.degree(static_cast<vtx_t>(u));
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      const vtx_t v = *it;
      if (better_endpoint(static_cast<vtx_t>(u), v, du, g.degree(v), orient)) {
        out.neighbors[static_cast<std::size_t>(cursor[u]++)] = v;
      }
    }
  }

  // Ensure sorted adjacency (source may already be sorted; keep invariant).
  for (offset_t u = 0; u < out.num_vertices; ++u) {
    auto* b =
        out.neighbors.data() + out.offsets[static_cast<std::size_t>(u)];
    auto* e =
        out.neighbors.data() + out.offsets[static_cast<std::size_t>(u) + 1];
    std::sort(b, e);
  }
  return out;
}

TcResult tc_serial(const Graph& g, TcOrientMode orient, IntersectMode mode) {
  TcResult r;
  if (g.num_vertices == 0) {
    return r;
  }
  Graph directed = orient_graph(g, orient);
  r.triangles =
      count_on_oriented(directed, mode, &r.wedges_examined, &r.intersect_ops);
  return r;
}

TcResult tc_serial_on_oriented(const Graph& directed, IntersectMode mode) {
  TcResult r;
  r.triangles =
      count_on_oriented(directed, mode, &r.wedges_examined, &r.intersect_ops);
  return r;
}

TcValidation validate_tc(const Graph& g, const TcResult& r, TcOrientMode orient,
                         IntersectMode mode) {
  TcValidation out;
  auto err = [&](const std::string& msg) {
    out.ok = false;
    out.errors.push_back(msg);
  };

  // Independent recount with the other intersection mode as a cross-check when
  // possible; always re-run serial as the oracle for the same (orient, mode).
  const TcResult again = tc_serial(g, orient, mode);
  if (again.triangles != r.triangles) {
    err("triangle total mismatch vs serial recount: got " +
        std::to_string(r.triangles) + " expected " +
        std::to_string(again.triangles));
  }

  // Cross-check merge vs galloping when degrees allow (same orientation).
  const IntersectMode other = (mode == IntersectMode::Merge)
                                  ? IntersectMode::Galloping
                                  : IntersectMode::Merge;
  const TcResult cross = tc_serial(g, orient, other);
  if (cross.triangles != r.triangles) {
    err("merge/galloping triangle totals disagree");
  }

  // Brute-force O(∑ deg^2) check on tiny graphs only.
  constexpr offset_t kBruteMaxV = 64;
  constexpr offset_t kBruteMaxE = 512;
  if (g.num_vertices <= kBruteMaxV && g.num_edges <= kBruteMaxE) {
    std::uint64_t brute = 0;
    for (offset_t u = 0; u < g.num_vertices; ++u) {
      const auto* ub = g.neighbors_begin(static_cast<vtx_t>(u));
      const auto* ue = g.neighbors_end(static_cast<vtx_t>(u));
      for (const auto* it = ub; it != ue; ++it) {
        const vtx_t v = *it;
        if (v <= static_cast<vtx_t>(u)) {
          continue;  // each undirected edge once (requires symmetrized CSR)
        }
        brute += intersect_merge(ub, ue, g.neighbors_begin(v),
                                 g.neighbors_end(v));
      }
    }
    // Each triangle counted 3 times (once per undirected edge).
    if (brute % 3 != 0) {
      err("brute edge-intersection total not divisible by 3 (graph not "
          "symmetrized?)");
    } else if (brute / 3 != r.triangles) {
      err("brute-force triangle count mismatch: got " +
          std::to_string(r.triangles) + " expected " +
          std::to_string(brute / 3));
    }
  }

  return out;
}

}  // namespace pargraph
