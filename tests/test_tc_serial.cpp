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
  const fs::path results_csv = results_dir / "tc_sample_serial.csv";

  pargraph::BuildOptions opts;
  opts.make_undirected = true;
  opts.dedupe = true;
  opts.drop_self_loops = true;
  opts.sort_adjacency = true;

  pargraph::Graph g = pargraph::load_edgelist_text(el.string(), opts);
  expect(g.num_vertices == 7, "sample V == 7");
  expect(g.num_edges == 22, "sample E == 22 (symmetrized)");
  expect(!g.directed, "sample undirected");
  expect(pargraph::validate_csr(g).ok, "validate sample CSR");
  pargraph::save_csr_binary(g, csr_path.string());
  expect(fs::exists(csr_path), "wrote tc_sample.csr");

  g = pargraph::load_csr_binary(csr_path.string());

  {
    pargraph::TcResult r =
        pargraph::tc_serial(g, pargraph::TcOrientMode::Degree,
                            pargraph::IntersectMode::Merge);
    expect(r.triangles == 6, "degree+merge finds 6 triangles");
    expect(r.wedges_examined == g.num_edges / 2,
           "wedges == undirected edge count");
    const auto vr = pargraph::validate_tc(g, r, pargraph::TcOrientMode::Degree,
                                          pargraph::IntersectMode::Merge);
    expect(vr.ok, "validate degree+merge");
    for (const auto& e : vr.errors) {
      std::cerr << "  " << e << "\n";
    }
  }

  {
    pargraph::TcResult r =
        pargraph::tc_serial(g, pargraph::TcOrientMode::Id,
                            pargraph::IntersectMode::Galloping);
    expect(r.triangles == 6, "id+galloping finds 6 triangles");
    expect(pargraph::validate_tc(g, r, pargraph::TcOrientMode::Id,
                                 pargraph::IntersectMode::Galloping)
               .ok,
           "validate id+galloping");
  }

  // Friendship triangle + pendant from sample_undirected.el → 1 triangle.
  {
    const fs::path friend_el = find_tiny("sample_undirected.el");
    expect(!friend_el.empty(), "locate sample_undirected.el");
    if (!friend_el.empty()) {
      pargraph::Graph fg =
          pargraph::load_edgelist_text(friend_el.string(), opts);
      pargraph::TcResult r = pargraph::tc_serial(fg);
      expect(r.triangles == 1, "friendship sample has 1 triangle");
      expect(pargraph::validate_tc(fg, r).ok, "validate friendship TC");
    }
  }

  // Empty / edgeless graphs.
  {
    pargraph::Graph empty;
    empty.num_vertices = 0;
    empty.num_edges = 0;
    empty.offsets = {0};
    empty.sorted = true;
    expect(pargraph::tc_serial(empty).triangles == 0, "empty graph → 0");

    pargraph::Graph iso;
    iso.num_vertices = 3;
    iso.num_edges = 0;
    iso.offsets = {0, 0, 0, 0};
    iso.sorted = true;
    expect(pargraph::tc_serial(iso).triangles == 0, "isolated verts → 0");
  }

  // Timed sample run + results artifacts.
  {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = "tc";
    meta.variant = "serial";
    meta.orient = "degree";
    meta.intersect = "merge";
    meta.graph_path = csr_path.string();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;
    meta.seed = 1;
    meta.trials = 3;
    meta.load_time_ms = 0.0;

    pargraph::Timer ot;
    ot.start();
    pargraph::Graph oriented =
        pargraph::orient_graph(g, pargraph::TcOrientMode::Degree);
    ot.stop();
    meta.orient_time_ms = ot.elapsed_ms();

    std::vector<pargraph::TcTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_rate;
    std::uint64_t tris = 0;

    for (int pass = -1; pass < meta.trials; ++pass) {
      pargraph::Timer t;
      t.start();
      pargraph::TcResult r = pargraph::tc_serial_on_oriented(
          oriented, pargraph::IntersectMode::Merge);
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
      row.valid_ok =
          pargraph::validate_tc(g, r, pargraph::TcOrientMode::Degree,
                                pargraph::IntersectMode::Merge)
              .ok;
      expect(row.valid_ok, "per-trial serial validate");
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
    expect(fs::exists(results_csv), "wrote results CSV");
    expect(fs::exists(results_csv.string() + ".meta.json"),
           "wrote results meta JSON");
  }

  if (g_failures == 0) {
    std::cout << "\nAll serial TC tests passed.\n";
    return 0;
  }
  std::cerr << "\n" << g_failures << " assertion(s) failed.\n";
  return 1;
}
