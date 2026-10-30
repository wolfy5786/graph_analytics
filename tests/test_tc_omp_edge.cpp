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

}  // namespace

int main() {
  const fs::path el = find_tiny("tc_sample.el");
  expect(!el.empty(), "locate tc_sample.el");
  if (el.empty()) {
    return 1;
  }

  const fs::path out_dir =
      fs::path(PARGRAPH_SOURCE_DIR) / "tests" / "tiny_graphs";
  const fs::path csr_path = out_dir / "tc_sample.csr";
  const fs::path results_dir = fs::path(PARGRAPH_SOURCE_DIR) / "results";
  fs::create_directories(results_dir);
  const fs::path results_csv = results_dir / "tc_sample_omp_edge.csv";

  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;

  pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
  expect(pargraph::validate_csr(g).ok, "validate sample CSR");
  pargraph::save_csr_binary(g, csr_path.string());
  g = pargraph::load_csr_binary(csr_path.string());

  const pargraph::Graph oriented =
      pargraph::orient_graph(g, pargraph::TcOrientMode::Degree);
  const pargraph::TcResult serial = pargraph::tc_serial_on_oriented(
      oriented, pargraph::IntersectMode::Merge);
  expect(serial.triangles == 6, "serial oracle = 6");

  const int thread_counts[] = {1, 2, 4, omp_get_max_threads()};
  for (int nt : thread_counts) {
    if (nt < 1) {
      continue;
    }
    const pargraph::TcResult edge_r =
        pargraph::tc_omp_edge(oriented, pargraph::IntersectMode::Merge, nt);
    expect(edge_r.triangles == serial.triangles,
           "omp_edge matches serial threads=" + std::to_string(nt));
    expect(edge_r.wedges_examined == serial.wedges_examined,
           "omp_edge wedges match serial threads=" + std::to_string(nt));
    expect(pargraph::validate_tc(g, edge_r, pargraph::TcOrientMode::Degree,
                                 pargraph::IntersectMode::Merge)
               .ok,
           "validate omp_edge threads=" + std::to_string(nt));
  }

  // Cross-check vs vertex-parallel design.
  {
    const pargraph::TcResult a =
        pargraph::tc_omp(oriented, pargraph::IntersectMode::Merge, 4);
    const pargraph::TcResult b =
        pargraph::tc_omp_edge(oriented, pargraph::IntersectMode::Merge, 4);
    expect(a.triangles == b.triangles, "omp vs omp_edge agree");
  }

  // Timed sample + results artifacts.
  {
    const int nt = std::max(2, omp_get_max_threads());
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "tc";
    meta.variant = "omp_edge";
    meta.orient = "degree";
    meta.intersect = "merge";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 42;
    meta.trials = 3;
    meta.threads = nt;
    meta.load_time_ms = 0.0;
    meta.orient_time_ms = 0.0;

    std::vector<pargraph::TcTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_rate;
    std::uint64_t tris = 0;

    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::TcResult r =
          pargraph::tc_omp_edge(oriented, pargraph::IntersectMode::Merge, nt);
      t.stop();
      pargraph::TcTrialRow row;
      row.trial = pass;
      row.is_warmup = (pass < 0);
      row.time_ms = t.elapsed_ms();
      row.triangles = r.triangles;
      row.wedges_examined = r.wedges_examined;
      row.intersect_ops = r.intersect_ops;
      row.intersect_rate =
          (t.elapsed_sec() > 0)
              ? static_cast<double>(r.intersect_ops) / t.elapsed_sec()
              : 0.0;
      row.validated = true;
      row.valid_ok = (r.triangles == serial.triangles);
      expect(row.valid_ok, "per-trial omp_edge validate");
      rows.push_back(row);
      tris = r.triangles;
      if (!row.is_warmup) {
        timed_ms.push_back(row.time_ms);
        timed_rate.push_back(row.intersect_rate);
      }
    }

    pargraph::write_tc_results_csv(results_csv.string(), meta, rows);
    const double med = pargraph::median_sorted(timed_ms);
    const double med_rate = pargraph::median_sorted(timed_rate);
    double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
    double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
    pargraph::write_tc_results_meta_json(results_csv.string() + ".meta.json",
                                         meta, med, min_ms, max_ms, tris,
                                         med_rate, timed_ms.size());
    expect(fs::exists(results_csv), "wrote omp_edge sample results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote omp_edge sample results meta JSON");
  }

  if (g_failures == 0) {
    std::cout << "\nAll OpenMP edge-parallel TC tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
