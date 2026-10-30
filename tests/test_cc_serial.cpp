#include "graph.hpp"
#include "kernels.hpp"
#include "metadata.hpp"
#include "results.hpp"
#include "timer.hpp"

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
  const fs::path el = find_tiny("cc_sample.el");
  expect(!el.empty(), "locate cc_sample.el");
  if (el.empty()) {
    return 1;
  }

  const fs::path out_dir =
      fs::path(PARGRAPH_SOURCE_DIR) / "tests" / "tiny_graphs";
  const fs::path csr_path = out_dir / "cc_sample.csr";
  const fs::path results_dir = fs::path(PARGRAPH_SOURCE_DIR) / "results";
  fs::create_directories(results_dir);
  const fs::path results_csv = results_dir / "cc_sample_serial.csv";

  // Convert sample edgelist -> undirected CSR binary.
  {
    pargraph::BuildOptions opts;
    opts.make_undirected = true;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;

    pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
    expect(g.num_vertices == 6, "sample V == 6");
    expect(g.num_edges == 10, "sample E == 10 (symmetrized)");
    expect(!g.directed, "sample undirected");
    expect(pargraph::validate_csr(g).ok, "validate sample CSR");

    pargraph::save_csr_binary(g, csr_path.string());
    expect(fs::exists(csr_path), "wrote cc_sample.csr");
  }

  // Load binary and run serial union-find CC.
  pargraph::Graph g = pargraph::load_csr_binary(csr_path.string());
  expect(pargraph::graphs_equal(
             g, pargraph::load_csr_binary(csr_path.string())),
         "reload stable");

  {
    pargraph::Timer t;
    t.start();
    pargraph::CcResult r = pargraph::cc_serial_unionfind(g);
    t.stop();
    expect(r.num_components == 2, "CC finds 2 components");
    expect(r.largest_component == 4, "largest component size == 4");
    expect(r.label[0] == r.label[1] && r.label[1] == r.label[2] &&
               r.label[2] == r.label[3],
           "vertices 0..3 same component");
    expect(r.label[4] == r.label[5], "vertices 4..5 same component");
    expect(r.label[0] != r.label[4], "components A and B are distinct");
    const auto vr = pargraph::validate_cc(g, r);
    expect(vr.ok, "partition validate CC");
    for (const auto& e : vr.errors) {
      std::cerr << "  " << e << "\n";
    }
    expect(t.elapsed_ms() >= 0.0, "timer ran for CC");
  }

  // Also check single-component bfs_sample if present.
  {
    const fs::path bfs_el = find_tiny("bfs_sample.el");
    if (!bfs_el.empty()) {
      pargraph::BuildOptions opts;
      opts.make_undirected = true;
      pargraph::Graph bg = pargraph::load_edgelist_text(bfs_el.string(), opts);
      pargraph::CcResult r = pargraph::cc_serial_unionfind(bg);
      expect(r.num_components == 1, "bfs_sample is one component");
      expect(r.largest_component == 6, "bfs_sample largest == 6");
      expect(pargraph::validate_cc(bg, r).ok, "validate bfs_sample CC");
    }
  }

  // Store results with metadata (warmup + timed trials).
  {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "cc";
    meta.variant = "serial";
    meta.algo = "unionfind";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 1;
    meta.trials = 3;
    meta.load_time_ms = 0.0;

    std::vector<pargraph::CcTrialRow> rows;
    std::vector<double> timed_ms;
    std::uint64_t comps = 0;
    std::uint64_t largest = 0;

    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::CcResult r = pargraph::cc_serial_unionfind(g);
      t.stop();
      pargraph::CcTrialRow row;
      row.trial = pass;
      row.is_warmup = (pass < 0);
      row.time_ms = t.elapsed_ms();
      row.num_components = r.num_components;
      row.largest_component = r.largest_component;
      row.edges_united = r.edges_united;
      row.validated = true;
      row.valid_ok = pargraph::validate_cc(g, r).ok;
      expect(row.valid_ok, "per-trial validate");
      rows.push_back(row);
      comps = r.num_components;
      largest = r.largest_component;
      if (!row.is_warmup) {
        timed_ms.push_back(row.time_ms);
      }
    }

    pargraph::write_cc_results_csv(results_csv.string(), meta, rows);
    const double med = pargraph::median_sorted(timed_ms);
    double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
    double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
    pargraph::write_cc_results_meta_json(results_csv.string() + ".meta.json",
                                         meta, med, min_ms, max_ms, comps,
                                         largest, timed_ms.size());
    expect(fs::exists(results_csv), "wrote results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote results meta JSON");
  }

  // Second pass: load CSR again and run CC (simulates CLI reuse).
  {
    pargraph::Graph g2 = pargraph::load_csr_binary(csr_path.string());
    pargraph::CcResult r = pargraph::cc_serial_unionfind(g2);
    expect(r.num_components == 2 && r.largest_component == 4,
           "reload+CC still correct");
  }

  if (g_failures == 0) {
    std::cout << "\nAll serial CC tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
