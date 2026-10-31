// Micro-benchmark comparing shared-memory BFS variants on synthetic graphs.
// Writes CSVs under results/ for the design-doc tables. Not a CTest target.

#include "graph.hpp"
#include "kernels.hpp"
#include "metadata.hpp"
#include "results.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;

namespace {

pargraph::Graph make_grid(int side) {
  std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges;
  edges.reserve(static_cast<std::size_t>(side * side * 2));
  auto id = [side](int r, int c) {
    return static_cast<pargraph::vtx_t>(r * side + c);
  };
  for (int r = 0; r < side; ++r) {
    for (int c = 0; c < side; ++c) {
      if (c + 1 < side) {
        edges.emplace_back(id(r, c), id(r, c + 1));
      }
      if (r + 1 < side) {
        edges.emplace_back(id(r, c), id(r + 1, c));
      }
    }
  }
  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;
  return pargraph::build_csr_from_edges(
      std::move(edges), static_cast<pargraph::offset_t>(side * side), opts);
}

// Preferential-attachment style: each new vertex links to `k` existing ones.
pargraph::Graph make_ba(int n, int k, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<pargraph::vtx_t> bag;
  bag.reserve(static_cast<std::size_t>(n * k * 2));
  std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges;
  const int start = std::max(k + 1, 2);
  for (int i = 0; i < start; ++i) {
    for (int j = i + 1; j < start; ++j) {
      edges.emplace_back(static_cast<pargraph::vtx_t>(i),
                         static_cast<pargraph::vtx_t>(j));
      bag.push_back(static_cast<pargraph::vtx_t>(i));
      bag.push_back(static_cast<pargraph::vtx_t>(j));
    }
  }
  for (int v = start; v < n; ++v) {
    for (int t = 0; t < k; ++t) {
      std::uniform_int_distribution<std::size_t> dist(0, bag.size() - 1);
      const pargraph::vtx_t u = bag[dist(rng)];
      edges.emplace_back(static_cast<pargraph::vtx_t>(v), u);
      bag.push_back(static_cast<pargraph::vtx_t>(v));
      bag.push_back(u);
    }
  }
  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;
  return pargraph::build_csr_from_edges(
      std::move(edges), static_cast<pargraph::offset_t>(n), opts);
}

using BfsFn = pargraph::BfsResult (*)(const pargraph::Graph&, pargraph::vtx_t,
                                      int);

pargraph::BfsResult run_serial(const pargraph::Graph& g, pargraph::vtx_t src,
                               int) {
  return pargraph::bfs_serial(g, src);
}
pargraph::BfsResult run_omp(const pargraph::Graph& g, pargraph::vtx_t src,
                            int nt) {
  return pargraph::bfs_omp(g, src, nt);
}
pargraph::BfsResult run_atomic(const pargraph::Graph& g, pargraph::vtx_t src,
                               int nt) {
  return pargraph::bfs_omp_atomic(g, src, nt);
}
pargraph::BfsResult run_diropt(const pargraph::Graph& g, pargraph::vtx_t src,
                               int nt) {
  return pargraph::bfs_omp_diropt(g, src, nt, 15, 18, false);
}

struct Variant {
  const char* name;
  BfsFn fn;
  bool parallel;
};

void bench_graph(const std::string& tag, const pargraph::Graph& g,
                 const std::vector<pargraph::vtx_t>& sources, int threads,
                 int trials, const fs::path& results_dir) {
  const Variant variants[] = {
      {"serial", run_serial, false},
      {"omp", run_omp, true},
      {"omp_atomic", run_atomic, true},
      {"omp_diropt", run_diropt, true},
  };

  for (const Variant& var : variants) {
    const int nt = var.parallel ? threads : 1;
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "bfs";
    meta.variant = var.name;
    meta.graph_path = tag;
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 1;
    meta.sources = static_cast<int>(sources.size());
    meta.trials = trials;
    meta.threads = nt;

    std::vector<pargraph::BfsTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_teps;

    for (pargraph::vtx_t src : sources) {
      const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
      for (int pass = -1; pass < trials; ++pass) {
        pargraph::Timer t;
        t.start();
        pargraph::BfsResult r = var.fn(g, src, nt);
        t.stop();
        const bool ok = pargraph::validate_bfs(g, src, r).ok &&
                        r.dist == serial.dist && r.visited == serial.visited;
        if (!ok) {
          std::cerr << "VALIDATION FAIL " << tag << " " << var.name
                    << " source=" << src << "\n";
          std::exit(1);
        }
        pargraph::BfsTrialRow row;
        row.source = static_cast<int>(src);
        row.trial = pass;
        row.is_warmup = (pass < 0);
        row.time_ms = t.elapsed_ms();
        row.visited = r.visited;
        row.edges_examined = r.edges_examined;
        row.teps = (t.elapsed_sec() > 0.0)
                       ? static_cast<double>(r.edges_examined) / t.elapsed_sec()
                       : 0.0;
        row.validated = true;
        row.valid_ok = true;
        rows.push_back(row);
        if (!row.is_warmup) {
          timed_ms.push_back(row.time_ms);
          timed_teps.push_back(row.teps);
        }
      }
    }

    const fs::path csv =
        results_dir / (tag + "_bfs_" + var.name + ".csv");
    pargraph::write_bfs_results_csv(csv.string(), meta, rows);
    const double med = pargraph::median_sorted(timed_ms);
    const double hteps = pargraph::harmonic_mean(timed_teps);
    const double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
    const double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
    pargraph::write_bfs_results_meta_json(csv.string() + ".meta.json", meta,
                                          med, min_ms, max_ms, hteps,
                                          timed_ms.size());
    std::cout << tag << " " << var.name << " threads=" << nt
              << " median_ms=" << med << " hTEPS=" << hteps << " -> " << csv
              << "\n";
  }
}

}  // namespace

int main() {
  const fs::path results_dir = fs::path(PARGRAPH_SOURCE_DIR) / "results";
  fs::create_directories(results_dir);

  const int threads = std::max(1, omp_get_max_threads());
  constexpr int trials = 3;

  // Road-like: many shallow levels.
  {
    const pargraph::Graph grid = make_grid(256);  // 65,536 verts
    std::vector<pargraph::vtx_t> srcs = {0, 128 * 256 + 128, 255 * 256 + 255,
                                         10, 5000};
    bench_graph("grid256", grid, srcs, threads, trials, results_dir);
  }

  // Social-like: skewed degrees, few deep frontiers.
  {
    const pargraph::Graph ba = make_ba(50000, 4, /*seed=*/42);
    std::vector<pargraph::vtx_t> srcs = {0, 100, 1000, 25000, 49999};
    bench_graph("ba50k", ba, srcs, threads, trials, results_dir);
  }

  // Tiny sample (matches other unit-test CSVs; useful for smoke timing).
  {
    const fs::path el =
        fs::path(PARGRAPH_SOURCE_DIR) / "tests/tiny_graphs/bfs_sample.el";
    pargraph::BuildOptions opts;
    opts.make_undirected = true;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;
    const pargraph::Graph sample =
        pargraph::load_edgelist_text(el.string(), opts);
    std::vector<pargraph::vtx_t> srcs = {0, 3};
    bench_graph("bfs_sample", sample, srcs, threads, trials, results_dir);
  }

  std::cout << "BFS parallel micro-benchmarks complete.\n";
  return 0;
}
