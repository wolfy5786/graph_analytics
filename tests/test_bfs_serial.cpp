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
  const fs::path el = find_tiny("bfs_sample.el");
  expect(!el.empty(), "locate bfs_sample.el");
  if (el.empty()) {
    return 1;
  }

  const fs::path out_dir =
      fs::path(PARGRAPH_SOURCE_DIR) / "tests" / "tiny_graphs";
  const fs::path csr_path = out_dir / "bfs_sample.csr";
  const fs::path results_dir = fs::path(PARGRAPH_SOURCE_DIR) / "results";
  fs::create_directories(results_dir);
  const fs::path results_csv = results_dir / "bfs_sample_serial.csv";

  // Convert sample edgelist -> undirected CSR binary.
  {
    pargraph::BuildOptions opts;
    opts.make_undirected = true;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;

    pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
    expect(g.num_vertices == 6, "sample V == 6");
    expect(g.num_edges == 12, "sample E == 12 (symmetrized)");
    expect(!g.directed, "sample undirected");
    expect(pargraph::validate_csr(g).ok, "validate sample CSR");

    pargraph::save_csr_binary(g, csr_path.string());
    expect(fs::exists(csr_path), "wrote bfs_sample.csr");
  }

  // Load binary and run serial BFS.
  pargraph::Graph g = pargraph::load_csr_binary(csr_path.string());
  expect(pargraph::graphs_equal(
             g, pargraph::load_csr_binary(csr_path.string())),
         "reload stable");

  // Distances from 0.
  {
    pargraph::Timer t;
    t.start();
    pargraph::BfsResult r = pargraph::bfs_serial(g, 0);
    t.stop();
    expect(r.visited == 6, "BFS(0) reaches all 6");
    expect(r.dist[0] == 0 && r.dist[1] == 1 && r.dist[2] == 2 &&
               r.dist[3] == 3 && r.dist[4] == 1 && r.dist[5] == 2,
           "BFS(0) distances match hand values");
    const auto vr = pargraph::validate_bfs(g, 0, r);
    expect(vr.ok, "Graph500 validate BFS(0)");
    for (const auto& e : vr.errors) {
      std::cerr << "  " << e << "\n";
    }
    expect(t.elapsed_ms() >= 0.0, "timer ran for BFS(0)");
  }

  // Distances from 3.
  {
    pargraph::BfsResult r = pargraph::bfs_serial(g, 3);
    expect(r.dist[3] == 0 && r.dist[2] == 1 && r.dist[1] == 2 &&
               r.dist[5] == 2 && r.dist[0] == 3 && r.dist[4] == 3,
           "BFS(3) distances match hand values");
    expect(pargraph::validate_bfs(g, 3, r).ok, "Graph500 validate BFS(3)");
  }

  // Store results with metadata (warmup + timed trials), then re-run path.
  {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "bfs";
    meta.variant = "serial";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 42;
    meta.sources = 2;
    meta.trials = 2;
    meta.load_time_ms = 0.0;

    std::vector<pargraph::BfsTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_teps;
    const pargraph::vtx_t srcs[] = {0, 3};

    for (pargraph::vtx_t src : srcs) {
      for (int pass = -1; pass < meta.trials; ++pass) {
        pargraph::Timer t;
        t.start();
        pargraph::BfsResult r = pargraph::bfs_serial(g, src);
        t.stop();
        pargraph::BfsTrialRow row;
        row.source = static_cast<int>(src);
        row.trial = pass;
        row.is_warmup = (pass < 0);
        row.time_ms = t.elapsed_ms();
        row.visited = r.visited;
        row.edges_examined = r.edges_examined;
        row.teps = (t.elapsed_sec() > 0)
                       ? static_cast<double>(r.edges_examined) / t.elapsed_sec()
                       : 0.0;
        row.validated = true;
        row.valid_ok = pargraph::validate_bfs(g, src, r).ok;
        expect(row.valid_ok, "per-trial validate source=" + std::to_string(src));
        rows.push_back(row);
        if (!row.is_warmup) {
          timed_ms.push_back(row.time_ms);
          timed_teps.push_back(row.teps);
        }
      }
    }

    pargraph::write_bfs_results_csv(results_csv.string(), meta, rows);
    const double med = pargraph::median_sorted(timed_ms);
    const double hteps = pargraph::harmonic_mean(timed_teps);
    double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
    double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
    pargraph::write_bfs_results_meta_json(results_csv.string() + ".meta.json",
                                          meta, med, min_ms, max_ms, hteps,
                                          timed_ms.size());
    expect(fs::exists(results_csv), "wrote results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote results meta JSON");
  }

  // Second pass: load CSR again and run BFS (simulates CLI reuse).
  {
    pargraph::Graph g2 = pargraph::load_csr_binary(csr_path.string());
    pargraph::BfsResult r = pargraph::bfs_serial(g2, 0);
    expect(r.dist[3] == 3 && r.visited == 6, "reload+BFS still correct");
  }

  if (g_failures == 0) {
    std::cout << "\nAll serial BFS tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
