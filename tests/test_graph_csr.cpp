#include "graph.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void expect(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  } else {
    std::cout << "OK:   " << msg << "\n";
  }
}

fs::path find_tiny(const char* name) {
  const fs::path candidates[] = {
      fs::path("tests/tiny_graphs") / name,
      fs::path("../tests/tiny_graphs") / name,
      fs::path("../../tests/tiny_graphs") / name,
      fs::path(PARGRAPH_SOURCE_DIR) / "tests/tiny_graphs" / name,
  };
  for (const auto& p : candidates) {
    if (fs::exists(p)) {
      return p;
    }
  }
  return {};
}

}  // namespace

int main() {
  const fs::path directed_path = find_tiny("sample_directed.el");
  const fs::path undirected_path = find_tiny("sample_undirected.el");
  expect(!directed_path.empty(), "locate sample_directed.el");
  expect(!undirected_path.empty(), "locate sample_undirected.el");
  if (directed_path.empty() || undirected_path.empty()) {
    return 1;
  }

  // --- Directed sample: drop loop + dedupe ---
  {
    pargraph::BuildOptions opts;
    opts.make_undirected = false;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;

    pargraph::Graph g =
        pargraph::load_edgelist_text(directed_path.string(), opts);
    expect(g.num_vertices == 5, "directed V == 5");
    expect(g.num_edges == 6, "directed E == 6 after drop/dedupe");
    expect(g.directed, "directed flag set");
    expect(g.sorted, "sorted flag set");
    expect(g.degree(0) == 2, "deg(0) == 2");
    expect(g.degree(1) == 1, "deg(1) == 1");
    expect(g.degree(2) == 1, "deg(2) == 1 (self-loop dropped)");
    expect(g.degree(3) == 1, "deg(3) == 1");
    expect(g.degree(4) == 1, "deg(4) == 1");

    const auto vr = pargraph::validate_csr(g);
    expect(vr.ok, "validate directed CSR");
    for (const auto& e : vr.errors) {
      std::cerr << "  validation: " << e << "\n";
    }

    const fs::path bin = fs::temp_directory_path() / "pargraph_sample_directed.csr";
    pargraph::save_csr_binary(g, bin.string());
    expect(fs::exists(bin), "wrote directed binary CSR");

    pargraph::Graph g2 = pargraph::load_csr_binary(bin.string());
    expect(pargraph::graphs_equal(g, g2), "directed round-trip equality");
    const auto vr2 = pargraph::validate_csr(g2);
    expect(vr2.ok, "validate loaded directed CSR");

    // Neighbor membership checks on loaded graph.
    auto has_edge = [&](pargraph::vtx_t u, pargraph::vtx_t v) {
      const auto* b = g2.neighbors_begin(u);
      const auto* e = g2.neighbors_end(u);
      return std::find(b, e, v) != e;
    };
    expect(has_edge(0, 1) && has_edge(0, 2), "loaded edges from 0");
    expect(has_edge(4, 0), "loaded edge 4->0");
    expect(!has_edge(2, 2), "self-loop absent after load");

    fs::remove(bin);
  }

  // --- Undirected sample: symmetrize ---
  {
    pargraph::BuildOptions opts;
    opts.make_undirected = true;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;

    pargraph::Graph g =
        pargraph::load_edgelist_text(undirected_path.string(), opts);
    expect(g.num_vertices == 4, "undirected V == 4");
    expect(g.num_edges == 8, "undirected E == 8 after symmetrize");
    expect(!g.directed, "undirected flag clear");

    const auto vr = pargraph::validate_csr(g);
    expect(vr.ok, "validate undirected CSR (symmetry)");
    for (const auto& e : vr.errors) {
      std::cerr << "  validation: " << e << "\n";
    }

    const fs::path bin =
        fs::temp_directory_path() / "pargraph_sample_undirected.csr";
    pargraph::save_csr_binary(g, bin.string());
    pargraph::Graph g2 = pargraph::load_csr_binary(bin.string());
    expect(pargraph::graphs_equal(g, g2), "undirected round-trip equality");
    expect(pargraph::validate_csr(g2).ok, "validate loaded undirected CSR");
    fs::remove(bin);
  }

  // --- In-memory build path ---
  {
    std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
        {0, 1}, {1, 0}, {1, 2}};
    pargraph::BuildOptions opts;
    opts.make_undirected = false;
    opts.dedupe = true;
    opts.sort_adjacency = true;
    pargraph::Graph g = pargraph::build_csr_from_edges(edges, 3, opts);
    expect(g.num_vertices == 3 && g.num_edges == 3, "in-memory build sizes");
    expect(pargraph::validate_csr(g).ok, "validate in-memory CSR");
  }

  if (g_failures == 0) {
    std::cout << "\nAll graph CSR tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
