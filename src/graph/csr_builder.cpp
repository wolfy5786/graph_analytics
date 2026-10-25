#include "graph.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pargraph {
namespace {

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error(msg);
}

void apply_edge_transforms(std::vector<std::pair<vtx_t, vtx_t>>& edges,
                           const BuildOptions& opts) {
  if (opts.drop_self_loops) {
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                               [](const auto& e) { return e.first == e.second; }),
                edges.end());
  }

  if (opts.make_undirected) {
    const offset_t n = edges.size();
    edges.reserve(n * 2);
    for (offset_t i = 0; i < n; ++i) {
      edges.emplace_back(edges[i].second, edges[i].first);
    }
  }

  if (opts.dedupe || opts.sort_adjacency) {
    std::sort(edges.begin(), edges.end());
  }
  if (opts.dedupe) {
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  }
}

Graph csr_from_prepared_edges(const std::vector<std::pair<vtx_t, vtx_t>>& edges,
                              offset_t num_vertices,
                              const BuildOptions& opts) {
  if (num_vertices == 0) {
    for (const auto& e : edges) {
      const offset_t need =
          static_cast<offset_t>(std::max(e.first, e.second)) + 1;
      if (need > num_vertices) {
        num_vertices = need;
      }
    }
  }

  Graph g;
  g.num_vertices = num_vertices;
  g.num_edges = edges.size();
  g.directed = !opts.make_undirected;
  g.sorted = opts.sort_adjacency;
  g.offsets.assign(static_cast<std::size_t>(num_vertices) + 1, 0);
  g.neighbors.resize(static_cast<std::size_t>(g.num_edges));

  for (const auto& e : edges) {
    if (static_cast<offset_t>(e.first) >= num_vertices ||
        static_cast<offset_t>(e.second) >= num_vertices) {
      fail("edge endpoint out of range for declared vertex count");
    }
    ++g.offsets[static_cast<std::size_t>(e.first) + 1];
  }

  for (offset_t v = 0; v < num_vertices; ++v) {
    g.offsets[static_cast<std::size_t>(v) + 1] +=
        g.offsets[static_cast<std::size_t>(v)];
  }

  std::vector<offset_t> cursor = g.offsets;
  for (const auto& e : edges) {
    const offset_t pos = cursor[e.first]++;
    g.neighbors[static_cast<std::size_t>(pos)] = e.second;
  }

  if (opts.sort_adjacency) {
    for (offset_t v = 0; v < num_vertices; ++v) {
      auto* begin = g.neighbors.data() + g.offsets[static_cast<std::size_t>(v)];
      auto* end =
          g.neighbors.data() + g.offsets[static_cast<std::size_t>(v) + 1];
      std::sort(begin, end);
    }
    g.sorted = true;
  }

  return g;
}

}  // namespace

Graph build_csr_from_edges(std::vector<std::pair<vtx_t, vtx_t>> edges,
                           offset_t num_vertices,
                           const BuildOptions& opts) {
  apply_edge_transforms(edges, opts);
  return csr_from_prepared_edges(edges, num_vertices, opts);
}

Graph load_edgelist_text(const std::string& path, const BuildOptions& opts) {
  std::ifstream in(path);
  if (!in) {
    fail("failed to open edgelist: " + path);
  }

  std::vector<std::pair<vtx_t, vtx_t>> edges;
  edges.reserve(1 << 20);

  std::string line;
  offset_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    long long u = 0;
    long long v = 0;
    if (!(iss >> u >> v)) {
      fail("malformed edgelist at line " + std::to_string(line_no) + ": " +
           line);
    }
    if (u < 0 || v < 0 ||
        static_cast<unsigned long long>(u) >
            static_cast<unsigned long long>(
                std::numeric_limits<vtx_t>::max()) ||
        static_cast<unsigned long long>(v) >
            static_cast<unsigned long long>(
                std::numeric_limits<vtx_t>::max())) {
      fail("vertex id out of uint32 range at line " + std::to_string(line_no));
    }
    edges.emplace_back(static_cast<vtx_t>(u), static_cast<vtx_t>(v));
  }

  return build_csr_from_edges(std::move(edges), 0, opts);
}

void save_csr_binary(const Graph& g, const std::string& path) {
  if (g.offsets.size() != static_cast<std::size_t>(g.num_vertices) + 1) {
    fail("save_csr_binary: offsets size mismatch");
  }
  if (g.neighbors.size() != static_cast<std::size_t>(g.num_edges)) {
    fail("save_csr_binary: neighbors size mismatch");
  }
  if (!g.offsets.empty() && g.offsets.back() != g.num_edges) {
    fail("save_csr_binary: last offset != num_edges");
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    fail("failed to open CSR for write: " + path);
  }

  out.write(kCsrMagic, 8);
  const std::uint32_t version = kCsrVersion;
  std::uint32_t flags = 0;
  if (g.directed) {
    flags |= kFlagDirected;
  }
  if (g.sorted) {
    flags |= kFlagSorted;
  }
  const std::uint32_t vtx_bytes = 4;
  const std::uint32_t offset_bytes = 8;
  const std::uint64_t reserved[2] = {0, 0};

  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
  out.write(reinterpret_cast<const char*>(&g.num_vertices),
            sizeof(g.num_vertices));
  out.write(reinterpret_cast<const char*>(&g.num_edges), sizeof(g.num_edges));
  out.write(reinterpret_cast<const char*>(&vtx_bytes), sizeof(vtx_bytes));
  out.write(reinterpret_cast<const char*>(&offset_bytes), sizeof(offset_bytes));
  out.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));

  out.write(reinterpret_cast<const char*>(g.offsets.data()),
            static_cast<std::streamsize>(g.offsets.size() * sizeof(offset_t)));
  out.write(
      reinterpret_cast<const char*>(g.neighbors.data()),
      static_cast<std::streamsize>(g.neighbors.size() * sizeof(vtx_t)));

  if (!out) {
    fail("failed while writing CSR: " + path);
  }
}

Graph load_csr_binary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("failed to open CSR for read: " + path);
  }

  char magic[8] = {};
  in.read(magic, 8);
  if (!in || std::memcmp(magic, kCsrMagic, 8) != 0) {
    fail("bad CSR magic: " + path);
  }

  std::uint32_t version = 0;
  std::uint32_t flags = 0;
  offset_t num_vertices = 0;
  offset_t num_edges = 0;
  std::uint32_t vtx_bytes = 0;
  std::uint32_t offset_bytes = 0;
  std::uint64_t reserved[2] = {0, 0};

  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  in.read(reinterpret_cast<char*>(&flags), sizeof(flags));
  in.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));
  in.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
  in.read(reinterpret_cast<char*>(&vtx_bytes), sizeof(vtx_bytes));
  in.read(reinterpret_cast<char*>(&offset_bytes), sizeof(offset_bytes));
  in.read(reinterpret_cast<char*>(reserved), sizeof(reserved));

  if (!in) {
    fail("truncated CSR header: " + path);
  }
  if (version != kCsrVersion) {
    fail("unsupported CSR version " + std::to_string(version));
  }
  if (vtx_bytes != 4 || offset_bytes != 8) {
    fail("unexpected CSR width (want vtx=4 offset=8)");
  }

  Graph g;
  g.num_vertices = num_vertices;
  g.num_edges = num_edges;
  g.directed = (flags & kFlagDirected) != 0;
  g.sorted = (flags & kFlagSorted) != 0;
  g.offsets.resize(static_cast<std::size_t>(num_vertices) + 1);
  g.neighbors.resize(static_cast<std::size_t>(num_edges));

  in.read(reinterpret_cast<char*>(g.offsets.data()),
          static_cast<std::streamsize>(g.offsets.size() * sizeof(offset_t)));
  in.read(reinterpret_cast<char*>(g.neighbors.data()),
          static_cast<std::streamsize>(g.neighbors.size() * sizeof(vtx_t)));

  if (!in) {
    fail("truncated CSR body: " + path);
  }

  // Ensure no trailing garbage of unexpected length is required; ignore extras.
  return g;
}

ValidationResult validate_csr(const Graph& g) {
  ValidationResult r;

  auto err = [&](const std::string& msg) {
    r.ok = false;
    r.errors.push_back(msg);
  };

  if (g.offsets.size() != static_cast<std::size_t>(g.num_vertices) + 1) {
    err("offsets size != num_vertices+1");
  }
  if (g.neighbors.size() != static_cast<std::size_t>(g.num_edges)) {
    err("neighbors size != num_edges");
  }
  if (g.offsets.empty()) {
    return r;
  }
  if (g.offsets.front() != 0) {
    err("offsets[0] != 0");
  }
  if (g.offsets.back() != g.num_edges) {
    err("offsets[V] != num_edges");
  }

  for (offset_t v = 0; v < g.num_vertices; ++v) {
    const offset_t lo = g.offsets[static_cast<std::size_t>(v)];
    const offset_t hi = g.offsets[static_cast<std::size_t>(v) + 1];
    if (lo > hi) {
      err("offsets not non-decreasing at vertex " + std::to_string(v));
      break;
    }
    for (offset_t i = lo; i < hi; ++i) {
      const vtx_t n = g.neighbors[static_cast<std::size_t>(i)];
      if (static_cast<offset_t>(n) >= g.num_vertices) {
        err("neighbor id out of range at vertex " + std::to_string(v));
        return r;
      }
      if (g.sorted && i + 1 < hi &&
          g.neighbors[static_cast<std::size_t>(i)] >
              g.neighbors[static_cast<std::size_t>(i) + 1]) {
        err("adjacency not sorted at vertex " + std::to_string(v));
        return r;
      }
    }
  }

  if (!g.directed) {
    // Every edge u->v must have a reverse v->u.
    for (offset_t u = 0; u < g.num_vertices; ++u) {
      const offset_t lo = g.offsets[static_cast<std::size_t>(u)];
      const offset_t hi = g.offsets[static_cast<std::size_t>(u) + 1];
      for (offset_t i = lo; i < hi; ++i) {
        const vtx_t v = g.neighbors[static_cast<std::size_t>(i)];
        const auto* begin = g.neighbors_begin(v);
        const auto* end = g.neighbors_end(v);
        if (!std::binary_search(begin, end, static_cast<vtx_t>(u)) &&
            std::find(begin, end, static_cast<vtx_t>(u)) == end) {
          err("undirected graph missing reverse edge for " +
              std::to_string(u) + "->" + std::to_string(v));
          return r;
        }
      }
    }
  }

  return r;
}

bool graphs_equal(const Graph& a, const Graph& b) {
  return a.num_vertices == b.num_vertices && a.num_edges == b.num_edges &&
         a.directed == b.directed && a.sorted == b.sorted &&
         a.offsets == b.offsets && a.neighbors == b.neighbors;
}

}  // namespace pargraph
