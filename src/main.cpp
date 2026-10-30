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
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#ifdef PARGRAPH_HAS_MPI
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#endif

static int g_rank = 0;

#ifdef PARGRAPH_HAS_MPI
int run_bfs_mpi_atomic_tests();
#endif

namespace {

void usage(const char* argv0) {
  if (g_rank != 0) {
    return;
  }
  std::cerr
      << "Usage: " << argv0
      << " --kernel <bfs|cc|tc> --variant <serial|omp|omp_atomic|omp_edge|"
         "mpi_atomic>"
      << " --graph <path.csr> [options]\n"
      << "Options:\n"
      << "  --sources <n>     random BFS roots (default 1)\n"
      << "  --seed <n>        RNG seed for sources (default 1)\n"
      << "  --trials <n>      timed trials after warmup (default 1)\n"
      << "  --threads <n>     OpenMP threads per process (default: all cores;\n"
      << "                    mpi_atomic default is cores/ranks)\n"
      << "  --validate        correctness checks vs. serial invariants\n"
      << "  --output <csv>    write per-trial results CSV (+ .meta.json)\n"
      << "  --source <v>      fixed BFS source (disables random --sources)\n"
      << "  --algo <name>     CC algorithm (default: unionfind)\n"
      << "  --orient <degree|id>          TC edge orientation (default degree)\n"
      << "  --intersect <merge|galloping> TC set intersection (default merge)\n"
      << "TC variants: serial | omp (vertex-parallel) | omp_edge "
         "(edge-parallel).\n"
      << "mpi_atomic is hybrid MPI+OpenMP. Launch with mpiexec -n <ranks>.\n";
}

pargraph::TcOrientMode parse_orient(const std::string& s) {
  if (s == "degree") {
    return pargraph::TcOrientMode::Degree;
  }
  if (s == "id") {
    return pargraph::TcOrientMode::Id;
  }
  throw std::invalid_argument("unknown --orient (use degree|id)");
}

pargraph::IntersectMode parse_intersect(const std::string& s) {
  if (s == "merge") {
    return pargraph::IntersectMode::Merge;
  }
  if (s == "galloping") {
    return pargraph::IntersectMode::Galloping;
  }
  throw std::invalid_argument("unknown --intersect (use merge|galloping)");
}

pargraph::TcResult run_tc_variant(const pargraph::Graph& oriented,
                                  const std::string& variant,
                                  pargraph::IntersectMode mode, int threads) {
  if (variant == "omp") {
    return pargraph::tc_omp(oriented, mode, threads);
  }
  if (variant == "omp_edge") {
    return pargraph::tc_omp_edge(oriented, mode, threads);
  }
  return pargraph::tc_serial_on_oriented(oriented, mode);
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

pargraph::BfsResult run_bfs_variant(const pargraph::Graph& g,
                                    pargraph::vtx_t src,
                                    const std::string& variant,
                                    int threads,
                                    const pargraph::MpiVertexPart* part) {
  if (variant == "omp") {
    return pargraph::bfs_omp(g, src, threads);
  }
  if (variant == "omp_atomic") {
    return pargraph::bfs_omp_atomic(g, src, threads);
  }
#ifdef PARGRAPH_HAS_MPI
  if (variant == "mpi_atomic") {
    return pargraph::bfs_mpi_atomic(*part, src, threads);
  }
#else
  (void)part;
#endif
  return pargraph::bfs_serial(g, src);
}

#ifdef PARGRAPH_HAS_MPI
void mpi_barrier() {
  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed");
  }
}

int mpi_agree(int bad) {
  int any = 0;
  if (MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) !=
      MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allreduce failed");
  }
  return any;
}
#endif

int run_bfs(const pargraph::Graph& g, pargraph::RunMetadata& meta,
            int sources, int trials, std::uint64_t seed, bool has_fixed_source,
            pargraph::vtx_t fixed_source, bool do_validate,
            const std::string& output_path,
            const pargraph::MpiVertexPart* part) {
  meta.sources = has_fixed_source ? 1 : sources;

  const auto srcs =
      pick_sources(g, sources, seed, has_fixed_source, fixed_source);
  std::vector<pargraph::BfsTrialRow> rows;
  std::vector<double> timed_ms;
  std::vector<double> timed_teps;
  std::vector<double> timed_compute;
  std::vector<double> timed_comm;
  std::uint64_t bytes_sum = 0;
  const bool mpi_run = meta.variant == "mpi_atomic";

  auto record = [&](pargraph::vtx_t src, int trial, bool warmup,
                    const pargraph::Timer& t, const pargraph::BfsResult& result,
                    bool valid_ok) {
    if (g_rank != 0) {
      return;
    }
    pargraph::BfsTrialRow row;
    row.source = static_cast<int>(src);
    row.trial = trial;
    row.is_warmup = warmup;
    row.time_ms = t.elapsed_ms();
    row.visited = result.visited;
    row.edges_examined = result.edges_examined;
    row.teps = (t.elapsed_sec() > 0.0)
                   ? static_cast<double>(result.edges_examined) / t.elapsed_sec()
                   : 0.0;
    row.validated = do_validate;
    row.valid_ok = do_validate && valid_ok;
    row.compute_ms = result.mpi.compute_sec * 1e3;
    row.comm_ms = result.mpi.comm_sec * 1e3;
    row.bytes_sent = result.mpi.bytes_sent;
    rows.push_back(row);
    if (!warmup) {
      timed_ms.push_back(row.time_ms);
      timed_teps.push_back(row.teps);
      timed_compute.push_back(row.compute_ms);
      timed_comm.push_back(row.comm_ms);
      bytes_sum += row.bytes_sent;
      std::cout << "BFS source=" << src << " trial=" << trial
                << " time_ms=" << row.time_ms << " visited=" << row.visited
                << " edges=" << row.edges_examined << " TEPS=" << row.teps;
      if (mpi_run) {
        std::cout << " compute_ms=" << row.compute_ms
                  << " comm_ms=" << row.comm_ms
                  << " bytes_sent=" << row.bytes_sent;
      }
      std::cout << "\n";
      std::cout.flush();
    }
  };

  auto check_result = [&](pargraph::vtx_t src, int trial,
                          const pargraph::BfsResult& result, bool& ok) -> int {
    ok = true;
    int bad = 0;
    if (g_rank == 0 && do_validate) {
      const auto vr = pargraph::validate_bfs(g, src, result);
      ok = vr.ok;
      if (!vr.ok) {
        std::cerr << "BFS validate failed source=" << src << " trial=" << trial
                  << ":\n";
        for (const auto& e : vr.errors) {
          std::cerr << "  - " << e << "\n";
        }
        bad = 1;
      } else if (meta.variant != "serial") {
        const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
        if (result.dist != serial.dist || result.visited != serial.visited ||
            result.edges_examined != serial.edges_examined) {
          std::cerr << "BFS parallel vs serial mismatch source=" << src
                    << " trial=" << trial << "\n";
          ok = false;
          bad = 1;
        }
      }
    }
#ifdef PARGRAPH_HAS_MPI
    if (mpi_run) {
      bad = mpi_agree(bad);
    }
#endif
    return bad;
  };

  for (pargraph::vtx_t src : srcs) {
    // Warm-up (excluded from median / harmonic TEPS).
    {
#ifdef PARGRAPH_HAS_MPI
      if (mpi_run) {
        mpi_barrier();
      }
#endif
      pargraph::Timer t;
      t.start();
      pargraph::BfsResult warm =
          run_bfs_variant(g, src, meta.variant, meta.threads, part);
      t.stop();
      bool ok = true;
      if (check_result(src, -1, warm, ok) != 0) {
        return 1;
      }
      record(src, -1, true, t, warm, ok);
    }

    for (int trial = 0; trial < trials; ++trial) {
#ifdef PARGRAPH_HAS_MPI
      if (mpi_run) {
        mpi_barrier();
      }
#endif
      pargraph::Timer t;
      t.start();
      pargraph::BfsResult r =
          run_bfs_variant(g, src, meta.variant, meta.threads, part);
      t.stop();
      bool ok = true;
      if (check_result(src, trial, r, ok) != 0) {
        return 1;
      }
      record(src, trial, false, t, r, ok);
    }
  }

  if (g_rank != 0) {
    return 0;
  }

  double min_ms = timed_ms.empty()
                      ? 0.0
                      : *std::min_element(timed_ms.begin(), timed_ms.end());
  double max_ms = timed_ms.empty()
                      ? 0.0
                      : *std::max_element(timed_ms.begin(), timed_ms.end());
  const double med_ms = pargraph::median_sorted(timed_ms);
  const double h_teps = pargraph::harmonic_mean(timed_teps);

  const double med_compute = pargraph::median_sorted(timed_compute);
  const double med_comm = pargraph::median_sorted(timed_comm);
  std::cout << "Summary (excl. warmup): median_ms=" << med_ms
            << " min_ms=" << min_ms << " max_ms=" << max_ms
            << " harmonic_mean_TEPS=" << h_teps;
  if (mpi_run) {
    const double phase = med_compute + med_comm;
    const double share = phase > 0.0 ? med_comm / phase : 0.0;
    std::cout << " median_compute_ms=" << med_compute
              << " median_comm_ms=" << med_comm << " comm_share=" << share
              << " bytes_sent=" << bytes_sum
              << " distribution_ms=" << meta.distribution_time_ms;
  }
  std::cout << "\n";

  if (!output_path.empty()) {
    pargraph::write_bfs_results_csv(output_path, meta, rows);
    const std::string meta_path = output_path + ".meta.json";
    pargraph::write_bfs_results_meta_json(
        meta_path, meta, med_ms, min_ms, max_ms, h_teps, timed_ms.size(),
        med_compute, med_comm, bytes_sum);
    std::cout << "Wrote " << output_path << " and " << meta_path << "\n";
  }

  return 0;
}

int run_cc(const pargraph::Graph& g, pargraph::RunMetadata& meta, int trials,
           bool do_validate, const std::string& output_path) {
  if (meta.algo.empty()) {
    meta.algo = "unionfind";
  }
  if (meta.algo != "unionfind") {
    std::cerr << "only --algo unionfind is implemented for serial CC\n";
    return 2;
  }

  std::vector<pargraph::CcTrialRow> rows;
  std::vector<double> timed_ms;
  std::uint64_t last_comps = 0;
  std::uint64_t last_largest = 0;

  auto run_one = [&](int trial_id, bool warmup) -> int {
    pargraph::Timer t;
    t.start();
    pargraph::CcResult r = pargraph::cc_serial_unionfind(g);
    t.stop();

    pargraph::CcTrialRow row;
    row.trial = trial_id;
    row.is_warmup = warmup;
    row.time_ms = t.elapsed_ms();
    row.num_components = r.num_components;
    row.largest_component = r.largest_component;
    row.edges_united = r.edges_united;
    last_comps = r.num_components;
    last_largest = r.largest_component;

    if (do_validate) {
      row.validated = true;
      const auto vr = pargraph::validate_cc(g, r);
      row.valid_ok = vr.ok;
      if (!vr.ok) {
        std::cerr << "CC validate failed trial=" << trial_id << ":\n";
        for (const auto& e : vr.errors) {
          std::cerr << "  - " << e << "\n";
        }
        return 1;
      }
    }
    rows.push_back(row);

    if (!warmup) {
      timed_ms.push_back(row.time_ms);
      std::cout << "CC algo=" << meta.algo << " trial=" << trial_id
                << " time_ms=" << row.time_ms
                << " components=" << row.num_components
                << " largest=" << row.largest_component
                << " edges_united=" << row.edges_united << "\n";
    }
    return 0;
  };

  if (run_one(-1, true) != 0) {
    return 1;
  }
  for (int trial = 0; trial < trials; ++trial) {
    if (run_one(trial, false) != 0) {
      return 1;
    }
  }

  double min_ms = timed_ms.empty()
                      ? 0.0
                      : *std::min_element(timed_ms.begin(), timed_ms.end());
  double max_ms = timed_ms.empty()
                      ? 0.0
                      : *std::max_element(timed_ms.begin(), timed_ms.end());
  const double med_ms = pargraph::median_sorted(timed_ms);

  std::cout << "Summary (excl. warmup): median_ms=" << med_ms
            << " min_ms=" << min_ms << " max_ms=" << max_ms
            << " components=" << last_comps << " largest=" << last_largest
            << "\n";

  if (!output_path.empty()) {
    pargraph::write_cc_results_csv(output_path, meta, rows);
    const std::string meta_path = output_path + ".meta.json";
    pargraph::write_cc_results_meta_json(meta_path, meta, med_ms, min_ms,
                                         max_ms, last_comps, last_largest,
                                         timed_ms.size());
    std::cout << "Wrote " << output_path << " and " << meta_path << "\n";
  }

  return 0;
}

int run_tc(const pargraph::Graph& g, pargraph::RunMetadata& meta, int trials,
           bool do_validate, const std::string& output_path) {
  const pargraph::TcOrientMode orient = parse_orient(meta.orient);
  const pargraph::IntersectMode mode = parse_intersect(meta.intersect);

  if (meta.variant != "serial" && meta.variant != "omp" &&
      meta.variant != "omp_edge") {
    std::cerr << "TC supports --variant serial|omp|omp_edge\n";
    return 2;
  }

  pargraph::Timer orient_timer;
  orient_timer.start();
  pargraph::Graph oriented = pargraph::orient_graph(g, orient);
  orient_timer.stop();
  meta.orient_time_ms = orient_timer.elapsed_ms();

  std::vector<pargraph::TcTrialRow> rows;
  std::vector<double> timed_ms;
  std::vector<double> timed_rate;
  std::uint64_t last_triangles = 0;

  auto run_one = [&](int trial_id, bool warmup) -> int {
    pargraph::Timer t;
    t.start();
    pargraph::TcResult r =
        run_tc_variant(oriented, meta.variant, mode, meta.threads);
    t.stop();

    pargraph::TcTrialRow row;
    row.trial = trial_id;
    row.is_warmup = warmup;
    row.time_ms = t.elapsed_ms();
    row.triangles = r.triangles;
    row.wedges_examined = r.wedges_examined;
    row.intersect_ops = r.intersect_ops;
    row.intersect_rate =
        (t.elapsed_sec() > 0.0)
            ? static_cast<double>(r.intersect_ops) / t.elapsed_sec()
            : 0.0;
    last_triangles = r.triangles;

    if (do_validate) {
      row.validated = true;
      const auto vr = pargraph::validate_tc(g, r, orient, mode);
      row.valid_ok = vr.ok;
      if (!vr.ok) {
        std::cerr << "TC validate failed trial=" << trial_id << ":\n";
        for (const auto& e : vr.errors) {
          std::cerr << "  - " << e << "\n";
        }
        return 1;
      }
      if (meta.variant != "serial") {
        const pargraph::TcResult serial =
            pargraph::tc_serial_on_oriented(oriented, mode);
        if (serial.triangles != r.triangles) {
          std::cerr << "TC parallel vs serial mismatch trial=" << trial_id
                    << " got=" << r.triangles
                    << " expected=" << serial.triangles << "\n";
          return 1;
        }
      }
    }
    rows.push_back(row);

    if (!warmup) {
      timed_ms.push_back(row.time_ms);
      timed_rate.push_back(row.intersect_rate);
      std::cout << "TC variant=" << meta.variant << " orient=" << meta.orient
                << " intersect=" << meta.intersect << " trial=" << trial_id
                << " time_ms=" << row.time_ms
                << " triangles=" << row.triangles
                << " wedges=" << row.wedges_examined
                << " intersect_rate=" << row.intersect_rate << "\n";
    }
    return 0;
  };

  if (run_one(-1, true) != 0) {
    return 1;
  }
  for (int trial = 0; trial < trials; ++trial) {
    if (run_one(trial, false) != 0) {
      return 1;
    }
  }

  double min_ms = timed_ms.empty()
                      ? 0.0
                      : *std::min_element(timed_ms.begin(), timed_ms.end());
  double max_ms = timed_ms.empty()
                      ? 0.0
                      : *std::max_element(timed_ms.begin(), timed_ms.end());
  const double med_ms = pargraph::median_sorted(timed_ms);
  const double med_rate = pargraph::median_sorted(timed_rate);

  std::cout << "Summary (excl. warmup): median_ms=" << med_ms
            << " min_ms=" << min_ms << " max_ms=" << max_ms
            << " triangles=" << last_triangles
            << " median_intersect_rate=" << med_rate
            << " orient_ms=" << meta.orient_time_ms << "\n";

  if (!output_path.empty()) {
    pargraph::write_tc_results_csv(output_path, meta, rows);
    const std::string meta_path = output_path + ".meta.json";
    pargraph::write_tc_results_meta_json(meta_path, meta, med_ms, min_ms,
                                         max_ms, last_triangles, med_rate,
                                         timed_ms.size());
    std::cout << "Wrote " << output_path << " and " << meta_path << "\n";
  }

  return 0;
}

}  // namespace

int driver(int argc, char** argv) {
  std::string kernel;
  std::string variant = "serial";
  std::string graph_path;
  std::string output_path;
  std::string algo;
  std::string orient = "degree";
  std::string intersect = "merge";
  int sources = 1;
  int trials = 1;
  int threads = 0;  // 0 = OpenMP default
  std::uint64_t seed = 1;
  bool do_validate = false;
  bool has_fixed_source = false;
  pargraph::vtx_t fixed_source = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
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
    } else if (arg == "--algo") {
      algo = need("--algo");
    } else if (arg == "--orient") {
      orient = need("--orient");
    } else if (arg == "--intersect") {
      intersect = need("--intersect");
    } else if (arg == "--sources") {
      sources = std::stoi(need("--sources"));
    } else if (arg == "--source") {
      has_fixed_source = true;
      fixed_source = static_cast<pargraph::vtx_t>(std::stoul(need("--source")));
    } else if (arg == "--seed") {
      seed = std::stoull(need("--seed"));
    } else if (arg == "--trials") {
      trials = std::stoi(need("--trials"));
    } else if (arg == "--threads") {
      threads = std::stoi(need("--threads"));
    } else if (arg == "--validate") {
      do_validate = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
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
  if (kernel != "bfs" && kernel != "cc" && kernel != "tc") {
    std::cerr << "only --kernel bfs|cc|tc is implemented\n";
    return 2;
  }
  if (variant != "serial" && variant != "omp" && variant != "omp_atomic" &&
      variant != "omp_edge" && variant != "mpi_atomic") {
    if (g_rank == 0) {
      std::cerr << "only --variant serial|omp|omp_atomic|omp_edge|mpi_atomic "
                   "is implemented\n";
    }
    return 2;
  }
#ifndef PARGRAPH_HAS_MPI
  if (variant == "mpi_atomic") {
    if (g_rank == 0) {
      std::cerr << "mpi_atomic requires configuring with -DPARGRAPH_ENABLE_MPI=ON\n";
    }
    return 2;
  }
#endif
  if (kernel == "cc" && variant != "serial") {
    if (g_rank == 0) {
      std::cerr << "CC only supports --variant serial currently\n";
    }
    return 2;
  }
  if (kernel == "tc" && variant != "serial" && variant != "omp" &&
      variant != "omp_edge") {
    if (g_rank == 0) {
      std::cerr << "TC supports --variant serial|omp|omp_edge\n";
    }
    return 2;
  }
  if (kernel == "bfs" && variant == "omp_edge") {
    if (g_rank == 0) {
      std::cerr << "omp_edge is a TC variant only\n";
    }
    return 2;
  }
#ifdef PARGRAPH_HAS_MPI
  int world = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &world) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed");
  }
  if (world > 1 && variant != "mpi_atomic" && g_rank != 0) {
    return 0;
  }
#else
  const int world = 1;
#endif
  if (trials < 1) {
    trials = 1;
  }

  try {
    pargraph::RunMetadata meta = pargraph::make_run_metadata();
    meta.kernel = kernel;
    meta.variant = variant;
    meta.graph_path = graph_path;
    meta.seed = seed;
    meta.trials = trials;
    meta.algo = algo;
    meta.orient = orient;
    meta.intersect = intersect;
    // Validate TC flag values early (throws on bad input).
    if (kernel == "tc") {
      (void)parse_orient(orient);
      (void)parse_intersect(intersect);
    }
    if (variant == "mpi_atomic" && threads <= 0) {
      threads = std::max(1, omp_get_num_procs() / std::max(1, world));
    }
    if (variant == "omp" || variant == "omp_atomic" || variant == "omp_edge" ||
        variant == "mpi_atomic") {
      if (threads > 0) {
        omp_set_num_threads(threads);
      }
      meta.threads = threads > 0 ? threads : omp_get_max_threads();
    } else {
      meta.threads = 1;
    }
    meta.ranks = world;

    pargraph::Timer load_timer;
    load_timer.start();
    pargraph::Graph g = pargraph::load_csr_binary(graph_path);
    load_timer.stop();
    double load_ms = load_timer.elapsed_ms();
#ifdef PARGRAPH_HAS_MPI
    if (variant == "mpi_atomic") {
      double load_max = 0.0;
      if (MPI_Allreduce(&load_ms, &load_max, 1, MPI_DOUBLE, MPI_MAX,
                        MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce load time failed");
      }
      load_ms = load_max;
    }
#endif
    meta.load_time_ms = load_ms;
    meta.num_vertices = g.num_vertices;
    meta.num_edges = g.num_edges;
    meta.directed = g.directed;

    const auto csr_ok = pargraph::validate_csr(g);
    if (g_rank == 0 && !csr_ok.ok) {
      std::cerr << "CSR validation failed:\n";
      for (const auto& e : csr_ok.errors) {
        std::cerr << "  - " << e << "\n";
      }
    }
    if (!csr_ok.ok) {
      return 1;
    }

    pargraph::MpiVertexPart part;
#ifdef PARGRAPH_HAS_MPI
    if (variant == "mpi_atomic") {
      if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Barrier failed");
      }
      pargraph::Timer dist_timer;
      if (g_rank == 0) {
        dist_timer.start();
      }
      part = pargraph::mpi_partition_vertices_1d(g);
      if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Barrier failed");
      }
      if (g_rank == 0) {
        dist_timer.stop();
        meta.distribution_time_ms = dist_timer.elapsed_ms();
      }
    }
#endif

    if (g_rank == 0) {
      std::cout << "Loaded " << graph_path << " in " << meta.load_time_ms
                << " ms  V=" << g.num_vertices << " E=" << g.num_edges
                << " directed=" << (g.directed ? "yes" : "no")
                << " variant=" << variant << " threads=" << meta.threads
                << " ranks=" << meta.ranks;
      if (variant == "mpi_atomic") {
        std::cout << " distribution_ms=" << meta.distribution_time_ms
                  << " owned=[" << part.lo << "," << part.hi << ")";
      }
      std::cout << "\n";
    }

    if (kernel == "bfs") {
      return run_bfs(g, meta, sources, trials, seed, has_fixed_source,
                     fixed_source, do_validate, output_path, &part);
    }
    if (kernel == "cc") {
      return run_cc(g, meta, trials, do_validate, output_path);
    }
    return run_tc(g, meta, trials, do_validate, output_path);
  } catch (const std::invalid_argument& ex) {
    if (g_rank == 0) {
      std::cerr << ex.what() << "\n";
      usage(argv[0]);
    }
    return 2;
  } catch (const std::exception& ex) {
    if (g_rank == 0) {
      std::cerr << "error: " << ex.what() << "\n";
    }
    return 1;
  }
}

#ifdef PARGRAPH_HAS_MPI
int main(int argc, char** argv) {
  int provided = 0;
  if (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided) !=
      MPI_SUCCESS) {
    std::cerr << "MPI_Init_thread failed\n";
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  if (provided < MPI_THREAD_FUNNELED && g_rank == 0) {
    std::cerr << "warning: MPI provides less than MPI_THREAD_FUNNELED\n";
  }
  if (argc >= 2 && std::string(argv[1]) == "--mpi-self-test") {
    const int rc = run_bfs_mpi_atomic_tests();
    MPI_Finalize();
    return rc;
  }
  int rc = 1;
  try {
    rc = driver(argc, argv);
  } catch (const std::invalid_argument& ex) {
    if (g_rank == 0) {
      std::cerr << ex.what() << "\n";
    }
    rc = 2;
  } catch (const std::exception& ex) {
    if (g_rank == 0) {
      std::cerr << "error: " << ex.what() << "\n";
    }
    rc = 1;
  }
  MPI_Finalize();
  return rc;
}
#else
int main(int argc, char** argv) { return driver(argc, argv); }
#endif
