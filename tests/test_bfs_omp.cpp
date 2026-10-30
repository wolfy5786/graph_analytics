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
  const fs::path results_csv = results_dir / "bfs_sample_omp.csv";

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

  // Hand distances + serial oracle for sources 0 and 3.
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
      const pargraph::BfsResult omp_r = pargraph::bfs_omp(g, src, nt);
      expect(distances_match(omp_r, serial),
             "omp BFS(0) distances match serial threads=" +
                 std::to_string(nt));
      const auto vr = pargraph::validate_bfs(g, src, omp_r);
      expect(vr.ok, "Graph500 validate omp BFS(0) threads=" +
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
      const pargraph::BfsResult omp_r = pargraph::bfs_omp(g, src, nt);
      expect(distances_match(omp_r, serial),
             "omp BFS(3) distances match serial threads=" +
                 std::to_string(nt));
      expect(pargraph::validate_bfs(g, src, omp_r).ok,
             "Graph500 validate omp BFS(3) threads=" + std::to_string(nt));
    }
  }

  // Empty graph / out-of-range source.
  {
    pargraph::Graph empty;
    empty.num_vertices = 0;
    empty.num_edges = 0;
    empty.offsets = {0};
    empty.directed = true;
    empty.sorted = true;
    const pargraph::BfsResult r = pargraph::bfs_omp(empty, 0, 4);
    expect(r.visited == 0 && r.edges_examined == 0 && r.dist.empty(),
           "omp empty graph visits nothing");
    const pargraph::BfsResult bad = pargraph::bfs_omp(g, 999, 4);
    expect(bad.visited == 0 && bad.dist[0] == pargraph::kBfsUnreached,
           "omp out-of-range source leaves graph unreached");
  }

  // Isolated vertex (degree 0): visited=1, edges_examined=0.
  {
    pargraph::Graph iso;
    iso.num_vertices = 3;
    iso.num_edges = 0;
    iso.offsets = {0, 0, 0, 0};
    iso.directed = true;
    iso.sorted = true;
    const pargraph::BfsResult serial = pargraph::bfs_serial(iso, 1);
    const pargraph::BfsResult omp_r = pargraph::bfs_omp(iso, 1, 4);
    expect(distances_match(omp_r, serial), "omp isolated vertex matches serial");
    expect(omp_r.visited == 1 && omp_r.edges_examined == 0 &&
               omp_r.dist[1] == 0 && omp_r.parent[1] == 1,
           "omp isolated vertex counts");
  }

  // Multi-path diamond: 0->{1,2}, 1->3, 2->3. Deterministic parent of 3 is
  // min(1,2)=1; distances must match serial; parents valid via validate_bfs.
  {
    std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
        {0, 1}, {0, 2}, {1, 3}, {2, 3}};
    pargraph::BuildOptions dopts;
    dopts.make_undirected = false;
    dopts.dedupe = true;
    dopts.drop_self_loops = true;
    dopts.sort_adjacency = true;
    pargraph::Graph diamond =
        pargraph::build_csr_from_edges(std::move(edges), 4, dopts);
    const pargraph::BfsResult serial = pargraph::bfs_serial(diamond, 0);
    const pargraph::BfsResult a = pargraph::bfs_omp(diamond, 0, 8);
    const pargraph::BfsResult b = pargraph::bfs_omp(diamond, 0, 8);
    expect(distances_match(a, serial), "omp diamond distances match serial");
    expect(pargraph::validate_bfs(diamond, 0, a).ok,
           "omp diamond Graph500 parent/dist valid");
    expect(a.parent == b.parent && a.dist == b.dist,
           "omp diamond parents deterministic across runs");
    expect(a.parent[3] == 1, "omp diamond picks smallest parent for vertex 3");
  }

  // Timed sample run + results artifacts (mirrors serial test pattern).
  {
    const int nt = std::max(2, omp_get_max_threads());
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "bfs";
    meta.variant = "omp";
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
        pargraph::BfsResult r = pargraph::bfs_omp(g, src, nt);
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
               "per-trial omp validate source=" + std::to_string(src) +
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
    expect(fs::exists(results_csv), "wrote omp sample results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote omp sample results meta JSON");
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP BFS tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
