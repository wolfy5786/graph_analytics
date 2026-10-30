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

bool matches_serial(const pargraph::Graph& g, const pargraph::CcResult& r) {
  const pargraph::CcResult serial = pargraph::cc_serial_unionfind(g);
  return r.num_components == serial.num_components &&
         r.largest_component == serial.largest_component &&
         pargraph::cc_same_partition(r.label, serial.label) &&
         pargraph::validate_cc(g, r).ok;
}

void check_algo(const pargraph::Graph& g, const std::string& algo, int nt,
                const std::string& tag) {
  const pargraph::CcResult r = pargraph::cc_omp_atomic(g, nt, algo);
  expect(matches_serial(g, r), tag + " matches serial threads=" +
                                   std::to_string(nt) + " algo=" + algo);
  expect(r.rounds > 0, tag + " reports rounds");
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

  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;

  pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
  expect(g.num_vertices == 6, "sample V == 6");
  pargraph::save_csr_binary(g, csr_path.string());
  g = pargraph::load_csr_binary(csr_path.string());

  const int thread_counts[] = {1, 2, 4, omp_get_max_threads()};
  const char* algos[] = {"labelprop", "sv", "afforest"};

  for (const char* algo : algos) {
    for (int nt : thread_counts) {
      if (nt < 1) {
        continue;
      }
      check_algo(g, algo, nt, "sample");
      const pargraph::CcResult r = pargraph::cc_omp_atomic(g, nt, algo);
      expect(r.num_components == 2 && r.largest_component == 4,
             std::string(algo) + " sample aggregates threads=" +
                 std::to_string(nt));
    }
  }

  {
    std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 4}};
    pargraph::BuildOptions dopts;
    dopts.make_undirected = true;
    dopts.dedupe = true;
    dopts.sort_adjacency = true;
    pargraph::Graph two =
        pargraph::build_csr_from_edges(std::move(edges), 7, dopts);
    for (const char* algo : algos) {
      check_algo(two, algo, 4, "two cycles");
    }
  }

  // Timed artifacts for the default atomic design (labelprop).
  {
    const int nt = std::max(2, omp_get_max_threads());
    const fs::path results_csv = results_dir / "cc_sample_omp_atomic.csv";
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "cc";
    meta.variant = "omp_atomic";
    meta.algo = "labelprop";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 1;
    meta.trials = 3;
    meta.threads = nt;

    std::vector<pargraph::CcTrialRow> rows;
    std::vector<double> timed_ms;
    std::uint64_t comps = 0;
    std::uint64_t largest = 0;
    std::uint64_t rounds = 0;

    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::CcResult r = pargraph::cc_omp_atomic(g, nt, "labelprop");
      t.stop();
      pargraph::CcTrialRow row;
      row.trial = pass;
      row.is_warmup = (pass < 0);
      row.time_ms = t.elapsed_ms();
      row.num_components = r.num_components;
      row.largest_component = r.largest_component;
      row.edges_united = r.edges_united;
      row.rounds = r.rounds;
      row.validated = true;
      row.valid_ok = matches_serial(g, r);
      expect(row.valid_ok, "per-trial omp_atomic validate");
      rows.push_back(row);
      comps = r.num_components;
      largest = r.largest_component;
      rounds = r.rounds;
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
                                         largest, timed_ms.size(), rounds);
    expect(fs::exists(results_csv), "wrote omp_atomic sample CSV");
  }

  // Also dump SV / Afforest sample benches for the design-doc comparison.
  for (const char* algo : {"sv", "afforest"}) {
    const int nt = std::max(2, omp_get_max_threads());
    const fs::path results_csv =
        results_dir / (std::string("cc_sample_omp_atomic_") + algo + ".csv");
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "cc";
    meta.variant = "omp_atomic";
    meta.algo = algo;
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.trials = 3;
    meta.threads = nt;
    std::vector<pargraph::CcTrialRow> rows;
    std::vector<double> timed_ms;
    std::uint64_t comps = 0;
    std::uint64_t largest = 0;
    std::uint64_t rounds = 0;
    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::CcResult r = pargraph::cc_omp_atomic(g, nt, algo);
      t.stop();
      pargraph::CcTrialRow row;
      row.trial = pass;
      row.is_warmup = (pass < 0);
      row.time_ms = t.elapsed_ms();
      row.num_components = r.num_components;
      row.largest_component = r.largest_component;
      row.edges_united = r.edges_united;
      row.rounds = r.rounds;
      row.validated = true;
      row.valid_ok = matches_serial(g, r);
      rows.push_back(row);
      comps = r.num_components;
      largest = r.largest_component;
      rounds = r.rounds;
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
                                         largest, timed_ms.size(), rounds);
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP atomic CC tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
