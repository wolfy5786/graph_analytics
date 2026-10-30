#include "graph.hpp"
#include "kernels.hpp"
#include "metadata.hpp"
#include "results.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
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
  const fs::path el = find_tiny("pagerank_sample.el");
  expect(!el.empty(), "locate pagerank_sample.el");
  if (el.empty()) {
    return 1;
  }

  const fs::path out_dir =
      fs::path(PARGRAPH_SOURCE_DIR) / "tests" / "tiny_graphs";
  const fs::path csr_path = out_dir / "pagerank_sample.csr";
  const fs::path results_dir = fs::path(PARGRAPH_SOURCE_DIR) / "results";
  fs::create_directories(results_dir);
  const fs::path results_csv = results_dir / "pagerank_sample_serial.csv";

  {
    pargraph::BuildOptions opts;
    opts.make_undirected = false;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;

    pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
    expect(g.num_vertices == 5, "sample V == 5");
    expect(g.num_edges == 6, "sample E == 6 (deduped, no loops)");
    expect(g.directed, "sample directed");
    expect(pargraph::validate_csr(g).ok, "validate sample CSR");

    pargraph::save_csr_binary(g, csr_path.string());
    expect(fs::exists(csr_path), "wrote pagerank_sample.csr");
  }

  pargraph::Graph g = pargraph::load_csr_binary(csr_path.string());
  pargraph::Graph in_g = pargraph::build_reverse_csr(g);
  expect(in_g.num_edges == g.num_edges, "reverse CSR preserves E");
  expect(pargraph::validate_csr(in_g).ok, "validate reverse CSR");

  pargraph::PageRankParams params;
  params.damping = 0.85;
  params.tolerance = 1e-12;
  params.max_iters = 1000;

  {
    pargraph::Timer t;
    t.start();
    pargraph::PageRankResult r = pargraph::pagerank_serial(g, in_g, params);
    t.stop();

    expect(r.iterations > 0 && r.iterations <= params.max_iters,
           "serial iterations in range");
    expect(std::fabs(r.score_sum - 1.0) < 1e-6, "serial scores sum to ~1");
    expect(r.residual_l1 < params.tolerance || r.iterations == params.max_iters,
           "serial converged or hit max_iters");

    const double hand[] = {0.21514102539680685, 0.12143493579373046,
                           0.2246546312184993, 0.220956436535761,
                           0.21781297105520223};
    double l1 = 0.0;
    for (int i = 0; i < 5; ++i) {
      l1 += std::fabs(r.scores[static_cast<std::size_t>(i)] - hand[i]);
    }
    expect(l1 < 1e-9, "serial scores match hand-computed reference");

    const auto vr = pargraph::validate_pagerank(g, r, params);
    expect(vr.ok, "validate serial PageRank");
    for (const auto& e : vr.errors) {
      std::cerr << "  " << e << "\n";
    }
    expect(t.elapsed_ms() >= 0.0, "timer ran");
  }

  // 3-cycle: uniform stationary distribution.
  {
    std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
        {0, 1}, {1, 2}, {2, 0}};
    pargraph::BuildOptions opts;
    opts.make_undirected = false;
    pargraph::Graph cycle = pargraph::build_csr_from_edges(edges, 3, opts);
    pargraph::Graph cycle_in = pargraph::build_reverse_csr(cycle);
    pargraph::PageRankParams p;
    p.damping = 0.85;
    p.tolerance = 1e-12;
    p.max_iters = 100;
    pargraph::PageRankResult r = pargraph::pagerank_serial(cycle, cycle_in, p);
    expect(std::fabs(r.scores[0] - 1.0 / 3.0) < 1e-9 &&
               std::fabs(r.scores[1] - 1.0 / 3.0) < 1e-9 &&
               std::fabs(r.scores[2] - 1.0 / 3.0) < 1e-9,
           "3-cycle scores are uniform 1/3");
  }

  // Results CSV / meta (warmup + timed trials).
  {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "pagerank";
    meta.variant = "serial";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 1;
    meta.trials = 3;
    meta.damping = params.damping;
    meta.tolerance = 1e-6;
    meta.max_iters = 100;
    meta.load_time_ms = 0.0;

    pargraph::PageRankParams run_params = params;
    run_params.tolerance = meta.tolerance;
    run_params.max_iters = meta.max_iters;

    std::vector<pargraph::PageRankTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_ms_per_iter;
    int last_iters = 0;
    double last_residual = 0.0;

    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::PageRankResult r =
          pargraph::pagerank_serial(g, in_g, run_params);
      t.stop();
      pargraph::PageRankTrialRow row;
      row.trial = pass;
      row.is_warmup = (pass < 0);
      row.time_ms = t.elapsed_ms();
      row.iterations = r.iterations;
      row.residual_l1 = r.residual_l1;
      row.score_sum = r.score_sum;
      row.edges_examined = r.edges_examined;
      row.ms_per_iter =
          (r.iterations > 0) ? row.time_ms / r.iterations : 0.0;
      row.validated = true;
      row.valid_ok = pargraph::validate_pagerank(g, r, run_params).ok;
      expect(row.valid_ok, "per-trial validate");
      rows.push_back(row);
      if (!row.is_warmup) {
        timed_ms.push_back(row.time_ms);
        timed_ms_per_iter.push_back(row.ms_per_iter);
        last_iters = row.iterations;
        last_residual = row.residual_l1;
      }
    }

    pargraph::write_pagerank_results_csv(results_csv.string(), meta, rows);
    const double med = pargraph::median_sorted(timed_ms);
    const double med_iter = pargraph::median_sorted(timed_ms_per_iter);
    double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
    double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
    pargraph::write_pagerank_results_meta_json(
        results_csv.string() + ".meta.json", meta, med, min_ms, max_ms,
        med_iter, last_iters, last_residual, timed_ms.size());
    expect(fs::exists(results_csv), "wrote results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote results meta JSON");
  }

  if (g_failures == 0) {
    std::cout << "\nAll serial PageRank tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
