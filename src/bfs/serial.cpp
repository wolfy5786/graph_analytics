#include "kernels.hpp"

#include <algorithm>
#include <queue>
#include <string>

namespace pargraph {

BfsResult bfs_serial(const Graph& g, vtx_t source) {
  BfsResult r;
  r.dist.assign(static_cast<std::size_t>(g.num_vertices), kBfsUnreached);
  r.parent.assign(static_cast<std::size_t>(g.num_vertices), -1);

  if (static_cast<offset_t>(source) >= g.num_vertices) {
    return r;
  }

  std::queue<vtx_t> q;
  r.dist[source] = 0;
  r.parent[source] = static_cast<std::int32_t>(source);
  q.push(source);
  r.visited = 1;

  while (!q.empty()) {
    const vtx_t u = q.front();
    q.pop();
    r.edges_examined += g.degree(u);

    const auto* begin = g.neighbors_begin(u);
    const auto* end = g.neighbors_end(u);
    for (const auto* it = begin; it != end; ++it) {
      const vtx_t v = *it;
      if (r.dist[v] == kBfsUnreached) {
        r.dist[v] = r.dist[u] + 1;
        r.parent[v] = static_cast<std::int32_t>(u);
        q.push(v);
        ++r.visited;
      }
    }
  }

  return r;
}

BfsValidation validate_bfs(const Graph& g, vtx_t source, const BfsResult& r) {
  BfsValidation out;
  auto err = [&](const std::string& msg) {
    out.ok = false;
    out.errors.push_back(msg);
  };

  if (r.dist.size() != static_cast<std::size_t>(g.num_vertices) ||
      r.parent.size() != static_cast<std::size_t>(g.num_vertices)) {
    err("dist/parent size mismatch with graph");
    return out;
  }
  if (static_cast<offset_t>(source) >= g.num_vertices) {
    err("source out of range");
    return out;
  }
  if (r.dist[source] != 0 ||
      r.parent[source] != static_cast<std::int32_t>(source)) {
    err("source must have dist=0 and parent=source");
  }

  std::uint64_t reached = 0;
  for (offset_t v = 0; v < g.num_vertices; ++v) {
    const std::int32_t d = r.dist[static_cast<std::size_t>(v)];
    const std::int32_t p = r.parent[static_cast<std::size_t>(v)];
    if (d == kBfsUnreached) {
      if (p != -1) {
        err("unreached vertex has non -1 parent");
        return out;
      }
      continue;
    }
    ++reached;
    if (d < 0) {
      err("negative distance");
      return out;
    }
    if (v == source) {
      continue;
    }
    if (p < 0 || static_cast<offset_t>(p) >= g.num_vertices) {
      err("invalid parent for vertex " + std::to_string(v));
      return out;
    }
    if (r.dist[static_cast<std::size_t>(p)] != d - 1) {
      err("parent not exactly one level closer at vertex " +
          std::to_string(v));
      return out;
    }
    const auto parent_v = static_cast<vtx_t>(p);
    const auto* nb = g.neighbors_begin(parent_v);
    const auto* ne = g.neighbors_end(parent_v);
    const auto want = static_cast<vtx_t>(v);
    const bool linked = g.sorted ? std::binary_search(nb, ne, want)
                                 : std::find(nb, ne, want) != ne;
    if (!linked) {
      err("parent has no stored edge to vertex " + std::to_string(v));
      return out;
    }
  }

  if (reached != r.visited) {
    err("visited count does not match reached set size");
  }

  // Edge relaxation: for every stored arc u->v with u reached, v is reached
  // and dist[v] <= dist[u] + 1. On undirected (symmetrized) graphs this is
  // equivalent to |dist[u]-dist[v]| <= 1.
  for (offset_t u = 0; u < g.num_vertices; ++u) {
    const std::int32_t du = r.dist[static_cast<std::size_t>(u)];
    if (du == kBfsUnreached) {
      continue;
    }
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      const vtx_t v = *it;
      const std::int32_t dv = r.dist[v];
      if (dv == kBfsUnreached) {
        err("reached u has unreached out-neighbor v: " + std::to_string(u) +
            "->" + std::to_string(v));
        return out;
      }
      if (dv > du + 1) {
        err("edge violates BFS relaxation: " + std::to_string(u) + "->" +
            std::to_string(v) + " dist " + std::to_string(du) + "/" +
            std::to_string(dv));
        return out;
      }
    }
  }

  return out;
}

}  // namespace pargraph
