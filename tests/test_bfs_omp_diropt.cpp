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
#include <utility>
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
  return a.dist == b.dist && a.visited == b.visited;
}

void check_against_serial(const pargraph::Graph& g, pargraph::vtx_t src,
                          int threads, int alpha, int beta,
                          const std::string& label) {
  const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
  const pargraph::BfsResult diropt =
      pargraph::bfs_omp_diropt(g, src, threads, alpha, beta, false);
  expect(distances_match(diropt, serial),
         label + " distances match serial threads=" + std::to_string(threads));
  const auto vr = pargraph::validate_bfs(g, src, diropt);
  expect(vr.ok, label + " Graph500 validate threads=" + std::to_string(threads));
  for (const auto& e : vr.errors) {
    std::cerr << "  " << e << "\n";
  }
}

// Small directed diamond: 0->1, 0->2, 1->3, 2->3. Forces reverse-CSR path.
pargraph::Graph make_directed_diamond() {
  std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
      {0, 1}, {0, 2}, {1, 3}, {2, 3}};
  pargraph::BuildOptions opts;
  opts.make_undirected = false;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;
  return pargraph::build_csr_from_edges(std::move(edges), 4, opts);
}

// Star: center 0 linked to 1..n-1. High scout at the source forces bottom-up
// under default GAPBS alpha when E >= alpha.
pargraph::Graph make_star(int n) {
  std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges;
  edges.reserve(static_cast<std::size_t>(n - 1));
  for (int i = 1; i < n; ++i) {
    edges.emplace_back(0, static_cast<pargraph::vtx_t>(i));
  }
  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;
  return pargraph::build_csr_from_edges(
      std::move(edges), static_cast<pargraph::offset_t>(n), opts);
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
  const fs::path results_csv = results_dir / "bfs_sample_omp_diropt.csv";

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
      check_against_serial(g, src, nt, 15, 18, "sample diropt BFS(0)");
    }
  }

  {
    const pargraph::vtx_t src = 3;
    for (int nt : thread_counts) {
      if (nt < 1) {
        continue;
      }
      check_against_serial(g, src, nt, 15, 18, "sample diropt BFS(3)");
    }
  }

  // Cross-check vs omp_atomic on the sample.
  {
    const pargraph::BfsResult a = pargraph::bfs_omp_atomic(g, 0, 4);
    const pargraph::BfsResult b = pargraph::bfs_omp_diropt(g, 0, 4);
    expect(a.dist == b.dist && a.visited == b.visited,
           "omp_atomic and omp_diropt agree on BFS(0) distances");
  }

  // Directed reverse-CSR path.
  {
    const pargraph::Graph diamond = make_directed_diamond();
    expect(diamond.directed, "diamond is directed");
    for (int nt : thread_counts) {
      if (nt < 1) {
        continue;
      }
      check_against_serial(diamond, 0, nt, 15, 18, "directed diamond diropt");
    }
    const pargraph::BfsResult r =
        pargraph::bfs_omp_diropt(diamond, 0, 2, 15, 18, false);
    expect(r.dist[0] == 0 && r.dist[1] == 1 && r.dist[2] == 1 &&
               r.dist[3] == 2,
           "directed diamond hand distances");
  }

  // Star from the center: scout = n-1 > (2(n-1))/alpha under default alpha=15.
  {
    const pargraph::Graph star = make_star(512);
    const pargraph::BfsResult r =
        pargraph::bfs_omp_diropt(star, 0, 4, /*alpha=*/15, /*beta=*/18,
                                 /*per_level=*/true);
    expect(pargraph::validate_bfs(star, 0, r).ok,
           "star diropt Graph500 validate");
    expect(distances_match(r, pargraph::bfs_serial(star, 0)),
           "star diropt matches serial");
    expect(!r.levels.empty(), "per-level stats populated");
    bool saw_bu = false;
    for (const auto& lv : r.levels) {
      if (lv.bottom_up) {
        saw_bu = true;
      }
    }
    expect(saw_bu, "high-degree source triggers bottom-up under alpha=15");
  }

  // Determinism of distances across repeated runs.
  {
    const pargraph::BfsResult once = pargraph::bfs_omp_diropt(g, 0, 2);
    const pargraph::BfsResult twice = pargraph::bfs_omp_diropt(g, 0, 2);
    expect(once.dist == twice.dist,
           "repeated diropt BFS(0) distances agree");
  }

  {
    const int nt = std::max(2, omp_get_max_threads());
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "bfs";
    meta.variant = "omp_diropt";
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
        pargraph::BfsResult r = pargraph::bfs_omp_diropt(g, src, nt);
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
               "per-trial omp_diropt validate source=" + std::to_string(src) +
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
    expect(fs::exists(results_csv), "wrote omp_diropt sample results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote omp_diropt sample results meta JSON");
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP direction-optimizing BFS tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
