#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pargraph {

using vtx_t = std::uint32_t;
using offset_t = std::uint64_t;

// Binary CSR layout (little-endian):
//   magic[8] = "PGRPHCSR"
//   version  u32 = 1
//   flags    u32  bit0=directed, bit1=sorted_adj
//   num_vertices u64
//   num_edges    u64
//   vtx_bytes    u32 (=4)
//   offset_bytes u32 (=8)
//   reserved     u64[2]
//   offsets[num_vertices+1] as u64
//   neighbors[num_edges]    as u32
inline constexpr char kCsrMagic[8] = {'P', 'G', 'R', 'P', 'H', 'C', 'S', 'R'};
inline constexpr std::uint32_t kCsrVersion = 1;
inline constexpr std::uint32_t kFlagDirected = 1u << 0;
inline constexpr std::uint32_t kFlagSorted = 1u << 1;

struct Graph {
  offset_t num_vertices = 0;
  offset_t num_edges = 0;
  std::vector<offset_t> offsets;   // size V+1
  std::vector<vtx_t> neighbors;    // size E
  bool directed = true;
  bool sorted = false;

  offset_t degree(vtx_t v) const {
    return offsets[static_cast<std::size_t>(v) + 1] -
           offsets[static_cast<std::size_t>(v)];
  }

  const vtx_t* neighbors_begin(vtx_t v) const {
    return neighbors.data() + offsets[static_cast<std::size_t>(v)];
  }

  const vtx_t* neighbors_end(vtx_t v) const {
    return neighbors.data() + offsets[static_cast<std::size_t>(v) + 1];
  }
};

struct BuildOptions {
  bool make_undirected = false;  // insert reverse edges
  bool dedupe = true;
  bool drop_self_loops = true;
  bool sort_adjacency = true;
};

struct ValidationResult {
  bool ok = true;
  std::vector<std::string> errors;
};

// Build CSR from an in-memory edgelist. If num_vertices == 0, infer max_id+1.
Graph build_csr_from_edges(std::vector<std::pair<vtx_t, vtx_t>> edges,
                           offset_t num_vertices,
                           const BuildOptions& opts);

// Load SNAP-style whitespace edgelist (# comments). Supports --format edgelist.
Graph load_edgelist_text(const std::string& path, const BuildOptions& opts);

// Persist / reload binary CSR.
void save_csr_binary(const Graph& g, const std::string& path);
Graph load_csr_binary(const std::string& path);

ValidationResult validate_csr(const Graph& g);

bool graphs_equal(const Graph& a, const Graph& b);

}  // namespace pargraph
