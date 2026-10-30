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
  const fs::path results_csv = results_dir / "pagerank_sample_omp_atomic.csv";

  {
    pargraph::BuildOptions opts;
    opts.make_undirected = false;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;
    pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
    pargraph::save_csr_binary(g, csr_path.string());
  }

  pargraph::Graph g = pargraph::load_csr_binary(csr_path.string());
  pargraph::Graph in_g = pargraph::build_reverse_csr(g);

  pargraph::PageRankParams params;
  params.damping = 0.85;
  params.tolerance = 1e-6;
  params.max_iters = 100;

  const pargraph::PageRankResult serial =
      pargraph::pagerank_serial(g, in_g, params);

  const int max_threads = omp_get_max_threads();
  // Push atomics reorder FP adds; compare within 1e-4 L1.
  constexpr double kPushTol = 1e-4;
  for (int nt = 1; nt <= max_threads; ++nt) {
    const pargraph::PageRankResult push_r =
        pargraph::pagerank_omp_atomic(g, params, nt);
    const auto vr =
        pargraph::validate_pagerank(g, push_r, params, &serial, kPushTol);
    expect(vr.ok, "omp_atomic push matches serial threads=" +
                      std::to_string(nt));
    for (const auto& e : vr.errors) {
      std::cerr << "  " << e << "\n";
    }
  }

  {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "pagerank";
    meta.variant = "omp_atomic";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.threads = max_threads;
    meta.seed = 1;
    meta.trials = 2;
    meta.damping = params.damping;
    meta.tolerance = params.tolerance;
    meta.max_iters = params.max_iters;

    std::vector<pargraph::PageRankTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_ms_per_iter;
    int last_iters = 0;
    double last_residual = 0.0;

    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::PageRankResult r =
          pargraph::pagerank_omp_atomic(g, params, max_threads);
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
      row.valid_ok =
          pargraph::validate_pagerank(g, r, params, &serial, kPushTol).ok;
      expect(row.valid_ok, "omp_atomic per-trial validate");
      rows.push_back(row);
      if (!row.is_warmup) {
        timed_ms.push_back(row.time_ms);
        timed_ms_per_iter.push_back(row.ms_per_iter);
        last_iters = row.iterations;
        last_residual = row.residual_l1;
      }
    }

    pargraph::write_pagerank_results_csv(results_csv.string(), meta, rows);
    pargraph::write_pagerank_results_meta_json(
        results_csv.string() + ".meta.json", meta,
        pargraph::median_sorted(timed_ms),
        *std::min_element(timed_ms.begin(), timed_ms.end()),
        *std::max_element(timed_ms.begin(), timed_ms.end()),
        pargraph::median_sorted(timed_ms_per_iter), last_iters, last_residual,
        timed_ms.size());
    expect(fs::exists(results_csv), "wrote omp_atomic results CSV");
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP push (atomic) PageRank tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
