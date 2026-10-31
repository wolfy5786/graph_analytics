#include "kernels.hpp"

#include <string>
#include <vector>

namespace pargraph {
namespace {

class UnionFind {
 public:
  explicit UnionFind(offset_t n) : parent_(static_cast<std::size_t>(n)),
                                   rank_(static_cast<std::size_t>(n), 0) {
    for (offset_t i = 0; i < n; ++i) {
      parent_[static_cast<std::size_t>(i)] = static_cast<vtx_t>(i);
    }
  }

  vtx_t find(vtx_t x) {
    vtx_t r = x;
    while (parent_[r] != r) {
      r = parent_[r];
    }
    // Path compression.
    while (parent_[x] != x) {
      const vtx_t next = parent_[x];
      parent_[x] = r;
      x = next;
    }
    return r;
  }

  bool unite(vtx_t a, vtx_t b) {
    a = find(a);
    b = find(b);
    if (a == b) {
      return false;
    }
    if (rank_[a] < rank_[b]) {
      parent_[a] = b;
    } else if (rank_[a] > rank_[b]) {
      parent_[b] = a;
    } else {
      parent_[b] = a;
      ++rank_[a];
    }
    return true;
  }

  const std::vector<vtx_t>& parents() const { return parent_; }

 private:
  std::vector<vtx_t> parent_;
  std::vector<std::uint8_t> rank_;
};

}  // namespace

CcResult cc_serial_unionfind(const Graph& g) {
  CcResult r;
  const offset_t n = g.num_vertices;
  r.label.assign(static_cast<std::size_t>(n), 0);
  if (n == 0) {
    return r;
  }

  UnionFind uf(n);
  for (offset_t u = 0; u < n; ++u) {
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      uf.unite(static_cast<vtx_t>(u), *it);
      ++r.edges_united;
    }
  }

  for (offset_t v = 0; v < n; ++v) {
    r.label[static_cast<std::size_t>(v)] = uf.find(static_cast<vtx_t>(v));
  }
  cc_fill_aggregates(r);
  return r;
}

void cc_fill_aggregates(CcResult& r) {
  const offset_t n = static_cast<offset_t>(r.label.size());
  r.num_components = 0;
  r.largest_component = 0;
  if (n == 0) {
    return;
  }
  std::vector<std::uint64_t> sizes(static_cast<std::size_t>(n), 0);
  for (offset_t v = 0; v < n; ++v) {
    ++sizes[r.label[static_cast<std::size_t>(v)]];
  }
  for (offset_t v = 0; v < n; ++v) {
    const std::uint64_t sz = sizes[static_cast<std::size_t>(v)];
    if (sz > 0) {
      ++r.num_components;
      if (sz > r.largest_component) {
        r.largest_component = sz;
      }
    }
  }
}

bool cc_same_partition(const std::vector<vtx_t>& a, const std::vector<vtx_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  const std::size_t n = a.size();
  // Labels are vertex ids in [0, n). Map each side's root to the other.
  std::vector<vtx_t> a_to_b(n, static_cast<vtx_t>(n));
  std::vector<vtx_t> b_to_a(n, static_cast<vtx_t>(n));
  for (std::size_t i = 0; i < n; ++i) {
    const vtx_t la = a[i];
    const vtx_t lb = b[i];
    if (la >= n || lb >= n) {
      return false;
    }
    if (a_to_b[la] == n) {
      a_to_b[la] = lb;
    } else if (a_to_b[la] != lb) {
      return false;
    }
    if (b_to_a[lb] == n) {
      b_to_a[lb] = la;
    } else if (b_to_a[lb] != la) {
      return false;
    }
  }
  return true;
}

CcValidation validate_cc(const Graph& g, const CcResult& r) {
  CcValidation out;
  auto err = [&](const std::string& msg) {
    out.ok = false;
    out.errors.push_back(msg);
  };

  if (r.label.size() != static_cast<std::size_t>(g.num_vertices)) {
    err("label size mismatch with graph");
    return out;
  }

  // Every stored arc must connect same-component endpoints.
  for (offset_t u = 0; u < g.num_vertices; ++u) {
    const vtx_t lu = r.label[static_cast<std::size_t>(u)];
    if (lu >= g.num_vertices) {
      err("label out of range at vertex " + std::to_string(u));
      return out;
    }
    const auto* begin = g.neighbors_begin(static_cast<vtx_t>(u));
    const auto* end = g.neighbors_end(static_cast<vtx_t>(u));
    for (const auto* it = begin; it != end; ++it) {
      const vtx_t v = *it;
      if (r.label[v] != lu) {
        err("edge spans components: " + std::to_string(u) + "->" +
            std::to_string(v));
        return out;
      }
    }
  }

  // Recompute component sizes from labels; compare aggregates.
  std::vector<std::uint64_t> sizes(static_cast<std::size_t>(g.num_vertices), 0);
  for (offset_t v = 0; v < g.num_vertices; ++v) {
    const vtx_t lab = r.label[static_cast<std::size_t>(v)];
    // Labels should be canonical roots: label[label[v]] == label[v].
    if (r.label[lab] != lab) {
      err("non-canonical label at vertex " + std::to_string(v));
      return out;
    }
    ++sizes[lab];
  }

  std::uint64_t comps = 0;
  std::uint64_t largest = 0;
  std::uint64_t sum = 0;
  for (offset_t v = 0; v < g.num_vertices; ++v) {
    const std::uint64_t sz = sizes[static_cast<std::size_t>(v)];
    if (sz > 0) {
      ++comps;
      sum += sz;
      if (sz > largest) {
        largest = sz;
      }
    }
  }

  if (sum != g.num_vertices) {
    err("component sizes do not sum to V");
  }
  if (comps != r.num_components) {
    err("num_components mismatch: got " + std::to_string(r.num_components) +
        " expected " + std::to_string(comps));
  }
  if (largest != r.largest_component) {
    err("largest_component mismatch: got " +
        std::to_string(r.largest_component) + " expected " +
        std::to_string(largest));
  }

  return out;
}

}  // namespace pargraph
