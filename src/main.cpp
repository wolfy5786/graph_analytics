#include "graph.hpp"
#include "kernels.hpp"
#include "metadata.hpp"
#include "results.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --kernel bfs --variant serial --graph <path.csr> [options]\n"
      << "Options:\n"
      << "  --sources <n>     random BFS roots (default 1)\n"
      << "  --seed <n>        RNG seed for sources (default 1)\n"
      << "  --trials <n>      timed trials per source after warmup (default 1)\n"
      << "  --validate        Graph500-style tree/level checks\n"
      << "  --output <csv>    write per-trial results CSV (+ .meta.json)\n"
      << "  --source <v>     fixed source (disables random --sources)\n";
}

std::vector<pargraph::vtx_t> pick_sources(const pargraph::Graph& g, int n,
                                          std::uint64_t seed,
                                          bool has_fixed,
                                          pargraph::vtx_t fixed) {
  if (has_fixed) {
    return {fixed};
  }
  n = std::max(1, n);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, g.num_vertices - 1);
  std::vector<pargraph::vtx_t> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out.push_back(static_cast<pargraph::vtx_t>(dist(rng)));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel;
  std::string variant = "serial";
  std::string graph_path;
  std::string output_path;
  int sources = 1;
  int trials = 1;
  std::uint64_t seed = 1;
  bool do_validate = false;
  bool has_fixed_source = false;
  pargraph::vtx_t fixed_source = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--kernel") {
      kernel = need("--kernel");
    } else if (arg == "--variant") {
      variant = need("--variant");
    } else if (arg == "--graph") {
      graph_path = need("--graph");
    } else if (arg == "--output") {
      output_path = need("--output");
    } else if (arg == "--sources") {
      sources = std::stoi(need("--sources"));
    } else if (arg == "--source") {
      has_fixed_source = true;
      fixed_source = static_cast<pargraph::vtx_t>(std::stoul(need("--source")));
    } else if (arg == "--seed") {
      seed = std::stoull(need("--seed"));
    } else if (arg == "--trials") {
      trials = std::stoi(need("--trials"));
    } else if (arg == "--validate") {
      do_validate = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else if (arg == "--threads") {
      (void)need("--threads");  // accepted for CLI compatibility; serial=1
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (kernel.empty() || graph_path.empty()) {
    usage(argv[0]);
    return 2;
  }
  if (kernel != "bfs") {
    std::cerr << "only --kernel bfs is implemented (serial Phase 1)\n";
    return 2;
  }
  if (variant != "serial") {
    std::cerr << "only --variant serial is implemented\n";
    return 2;
  }
  if (trials < 1) {
    trials = 1;
  }

  try {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = kernel;
    meta.variant = variant;
    meta.graph_path = graph_path;
    meta.seed = seed;
    meta.sources = has_fixed_source ? 1 : sources;
    meta.trials = trials;
    meta.threads = 1;

    pargraph::Timer load_timer;
    load_timer.start();
    pargraph::Graph g = pargraph::load_csr_binary(graph_path);
    load_timer.stop();
    meta.load_time_ms = load_timer.elapsed_ms();
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;

    const auto csr_ok = pargraph::validate_csr(g);
    if (!csr_ok.ok) {
      std::cerr << "CSR validation failed:\n";
      for (const auto& e : csr_ok.errors) {
        std::cerr << "  - " << e << "\n";
      }
      return 1;
    }

    std::cout << "Loaded " << graph_path << " in " << meta.load_time_ms
              << " ms  V=" << g.num_vertices << " E=" << g.num_edges
              << " directed=" << (g.directed ? "yes" : "no") << "\n";

    const auto srcs =
        pick_sources(g, sources, seed, has_fixed_source, fixed_source);
    std::vector<pargraph::BfsTrialRow> rows;
    std::vector<double> timed_ms;
    std::vector<double> timed_teps;

    for (pargraph::vtx_t src : srcs) {
      // Warm-up (excluded from median / harmonic TEPS).
      {
        pargraph::Timer t;
        t.start();
        pargraph::BfsResult warm = pargraph::bfs_serial(g, src);
        t.stop();

        pargraph::BfsTrialRow row;
        row.source = static_cast<int>(src);
        row.trial = -1;
        row.is_warmup = true;
        row.time_ms = t.elapsed_ms();
        row.visited = warm.visited;
        row.edges_examined = warm.edges_examined;
        row.teps = (t.elapsed_sec() > 0.0)
                       ? static_cast<double>(warm.edges_examined) /
                             t.elapsed_sec()
                       : 0.0;
        if (do_validate) {
          row.validated = true;
          const auto vr = pargraph::validate_bfs(g, src, warm);
          row.valid_ok = vr.ok;
          if (!vr.ok) {
            std::cerr << "BFS validate failed for source " << src << ":\n";
            for (const auto& e : vr.errors) {
              std::cerr << "  - " << e << "\n";
            }
            return 1;
          }
        }
        rows.push_back(row);
      }

      for (int trial = 0; trial < trials; ++trial) {
        pargraph::Timer t;
        t.start();
        pargraph::BfsResult r = pargraph::bfs_serial(g, src);
        t.stop();

        pargraph::BfsTrialRow row;
        row.source = static_cast<int>(src);
        row.trial = trial;
        row.is_warmup = false;
        row.time_ms = t.elapsed_ms();
        row.visited = r.visited;
        row.edges_examined = r.edges_examined;
        row.teps = (t.elapsed_sec() > 0.0)
                       ? static_cast<double>(r.edges_examined) / t.elapsed_sec()
                       : 0.0;
        if (do_validate) {
          row.validated = true;
          row.valid_ok = pargraph::validate_bfs(g, src, r).ok;
          if (!row.valid_ok) {
            std::cerr << "BFS validate failed timed trial source=" << src
                      << " trial=" << trial << "\n";
            return 1;
          }
        }
        rows.push_back(row);
        timed_ms.push_back(row.time_ms);
        timed_teps.push_back(row.teps);

        std::cout << "BFS source=" << src << " trial=" << trial
                  << " time_ms=" << row.time_ms << " visited=" << row.visited
                  << " edges=" << row.edges_examined << " TEPS=" << row.teps
                  << "\n";
      }
    }

    double min_ms = timed_ms.empty()
                        ? 0.0
                        : *std::min_element(timed_ms.begin(), timed_ms.end());
    double max_ms = timed_ms.empty()
                        ? 0.0
                        : *std::max_element(timed_ms.begin(), timed_ms.end());
    const double med_ms = pargraph::median_sorted(timed_ms);
    const double h_teps = pargraph::harmonic_mean(timed_teps);

    std::cout << "Summary (excl. warmup): median_ms=" << med_ms
              << " min_ms=" << min_ms << " max_ms=" << max_ms
              << " harmonic_mean_TEPS=" << h_teps << "\n";

    if (!output_path.empty()) {
      pargraph::write_bfs_results_csv(output_path, meta, rows);
      const std::string meta_path = output_path + ".meta.json";
      pargraph::write_bfs_results_meta_json(
          meta_path, meta, med_ms, min_ms, max_ms, h_teps, timed_ms.size());
      std::cout << "Wrote " << output_path << " and " << meta_path << "\n";
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
