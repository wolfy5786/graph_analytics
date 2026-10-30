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
  const fs::path results_csv = results_dir / "cc_sample_omp.csv";

  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;

  pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
  expect(g.num_vertices == 6, "sample V == 6");
  expect(g.num_edges == 10, "sample E == 10 (symmetrized)");
  expect(pargraph::validate_csr(g).ok, "validate sample CSR");
  pargraph::save_csr_binary(g, csr_path.string());
  g = pargraph::load_csr_binary(csr_path.string());

  const int thread_counts[] = {1, 2, 4, omp_get_max_threads()};
  for (int nt : thread_counts) {
    if (nt < 1) {
      continue;
    }
    const pargraph::CcResult r = pargraph::cc_omp(g, nt);
    expect(r.num_components == 2, "omp CC components==2 threads=" +
                                      std::to_string(nt));
    expect(r.largest_component == 4, "omp CC largest==4 threads=" +
                                         std::to_string(nt));
    expect(matches_serial(g, r),
           "omp CC matches serial partition threads=" + std::to_string(nt));
    expect(r.rounds > 0, "omp CC reports rounds threads=" + std::to_string(nt));
  }

  // Directed WCC: two arcs that form one weak component.
  {
    std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
        {0, 1}, {2, 1}, {3, 4}};
    pargraph::BuildOptions dopts;
    dopts.make_undirected = false;
    dopts.dedupe = true;
    dopts.drop_self_loops = true;
    dopts.sort_adjacency = true;
    pargraph::Graph dg =
        pargraph::build_csr_from_edges(std::move(edges), 5, dopts);
    const pargraph::CcResult r = pargraph::cc_omp(dg, 4);
    expect(matches_serial(dg, r), "omp directed WCC matches serial");
    expect(r.num_components == 2, "omp directed WCC has 2 components");
  }

  {
    pargraph::Graph empty;
    empty.offsets = {0};
    empty.sorted = true;
    const pargraph::CcResult r = pargraph::cc_omp(empty, 2);
    expect(r.num_components == 0 && r.label.empty(), "omp empty graph");
  }

  {
    const int nt = std::max(2, omp_get_max_threads());
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "cc";
    meta.variant = "omp";
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
      pargraph::CcResult r = pargraph::cc_omp(g, nt);
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
      expect(row.valid_ok, "per-trial omp validate pass=" + std::to_string(pass));
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
    expect(fs::exists(results_csv), "wrote omp sample results CSV");
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP CC tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
