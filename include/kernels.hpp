#pragma once

#include "graph.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pargraph {

inline constexpr std::int32_t kBfsUnreached =
    std::numeric_limits<std::int32_t>::max();

struct BfsResult {
  std::vector<std::int32_t> dist;    // kBfsUnreached if not reached
  std::vector<std::int32_t> parent;  // -1 if unreached; source parents itself
  std::uint64_t visited = 0;
  std::uint64_t edges_examined = 0;  // sum of degrees of dequeued vertices
};

// Serial queue-based level-synchronous BFS. Works on whatever orientation the
// CSR stores (directed out-edges or undirected symmetrized).
BfsResult bfs_serial(const Graph& g, vtx_t source);

struct BfsValidation {
  bool ok = true;
  std::vector<std::string> errors;
};

// Graph500-style tree / level checks against dist/parent.
BfsValidation validate_bfs(const Graph& g, vtx_t source, const BfsResult& r);

}  // namespace pargraph
