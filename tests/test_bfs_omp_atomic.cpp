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

#include <omp.h>

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

bool distances_match(const pargraph::BfsResult& a,
                     const pargraph::BfsResult& b) {
  return a.dist == b.dist && a.visited == b.visited &&
         a.edges_examined == b.edges_examined;
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
  const fs::path results_csv = results_dir / "bfs_sample_omp_atomic.csv";

  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;

  pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
  expect(g.num_vertices == 6, "sample V == 6");
  expect(g.num_edges == 12, "sample E == 12 (symmetrized)");
  expect(pargraph::validate_csr(g).ok, "validate sample CSR");
  pargraph::save_csr_binary(g, csr_path.string());

  g = pargraph::load_csr_binary(csr_path.string());

  const int thread_counts[] = {1, 2, 4, omp_get_max_threads()};

  {
    const pargraph::vtx_t src = 0;
    const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
    expect(serial.dist[0] == 0 && serial.dist[1] == 1 && serial.dist[2] == 2 &&
               serial.dist[3] == 3 && serial.dist[4] == 1 &&
               serial.dist[5] == 2,
           "serial BFS(0) hand distances");

    for (int nt : thread_counts) {
      if (nt < 1) {
        continue;
      }
      const pargraph::BfsResult omp_r = pargraph::bfs_omp_atomic(g, src, nt);
      expect(distances_match(omp_r, serial),
             "omp_atomic BFS(0) distances match serial threads=" +
                 std::to_string(nt));
      const auto vr = pargraph::validate_bfs(g, src, omp_r);
      expect(vr.ok, "Graph500 validate omp_atomic BFS(0) threads=" +
                        std::to_string(nt));
      for (const auto& e : vr.errors) {
        std::cerr << "  " << e << "\n";
      }
    }
  }

  {
    const pargraph::vtx_t src = 3;
    const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
    expect(serial.dist[3] == 0 && serial.dist[2] == 1 && serial.dist[1] == 2 &&
               serial.dist[5] == 2 && serial.dist[0] == 3 &&
               serial.dist[4] == 3,
           "serial BFS(3) hand distances");

    for (int nt : thread_counts) {
      if (nt < 1) {
        continue;
      }
      const pargraph::BfsResult omp_r = pargraph::bfs_omp_atomic(g, src, nt);
      expect(distances_match(omp_r, serial),
             "omp_atomic BFS(3) distances match serial threads=" +
                 std::to_string(nt));
      expect(pargraph::validate_bfs(g, src, omp_r).ok,
             "Graph500 validate omp_atomic BFS(3) threads=" +
                 std::to_string(nt));
    }
  }

  // Cross-check vs existing no-atomic omp on the sample.
  {
    const pargraph::BfsResult a = pargraph::bfs_omp(g, 0, 4);
    const pargraph::BfsResult b = pargraph::bfs_omp_atomic(g, 0, 4);
    expect(a.dist == b.dist && a.visited == b.visited,
           "omp and omp_atomic agree on BFS(0) distances");
  }

  {
    const int nt = std::max(2, omp_get_max_threads());
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "bfs";
    meta.variant = "omp_atomic";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 42;
    meta.sources = 2;
    meta.trials = 2;
    meta.threads = nt;
    meta.load_time_ms = 0.0;

    std::vector<pargraph::BfsTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_teps;
    const pargraph::vtx_t srcs[] = {0, 3};

    for (pargraph::vtx_t src : srcs) {
      const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
      for (int pass = -1; pass < meta.trials; ++pass) {
        pargraph::Timer t;
        t.start();
        pargraph::BfsResult r = pargraph::bfs_omp_atomic(g, src, nt);
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
        row.valid_ok = pargraph::validate_bfs(g, src, r).ok &&
                       distances_match(r, serial);
        expect(row.valid_ok,
               "per-trial omp_atomic validate source=" + std::to_string(src) +
                   " pass=" + std::to_string(pass));
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
    expect(fs::exists(results_csv), "wrote omp_atomic sample results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote omp_atomic sample results meta JSON");
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP atomic BFS tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
