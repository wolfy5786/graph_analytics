#include "graph.hpp"
#include "kernels.hpp"
#include "metadata.hpp"
#include "results.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;
int g_rank = 0;
int g_ranks = 1;

void expect(bool cond, const std::string& msg) {
  if (!cond) {
    if (g_rank == 0) {
      std::cerr << "FAIL: " << msg << "\n";
    }
    ++g_failures;
  } else if (g_rank == 0) {
    std::cout << "OK:   " << msg << "\n";
  }
}

void mpi_check(int rc, const char* what) {
  if (rc == MPI_SUCCESS) {
    return;
  }
  char buf[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  MPI_Error_string(rc, buf, &len);
  throw std::runtime_error(std::string(what) + ": " +
                           std::string(buf, static_cast<std::size_t>(len)));
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
  return a.dist == b.dist && a.visited == b.visited &&
         a.edges_examined == b.edges_examined;
}

void check_against_serial(const pargraph::Graph& g,
                          const pargraph::MpiVertexPart& part,
                          pargraph::vtx_t src, int threads,
                          const std::string& label) {
  const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
  const pargraph::BfsResult mpi_r =
      pargraph::bfs_mpi_atomic(part, src, threads);
  expect(distances_match(mpi_r, serial),
         label + " distances match serial threads=" + std::to_string(threads) +
             " ranks=" + std::to_string(g_ranks));
  const auto vr = pargraph::validate_bfs(g, src, mpi_r);
  expect(vr.ok, label + " Graph500 validate threads=" + std::to_string(threads) +
                    " ranks=" + std::to_string(g_ranks));
  if (g_rank == 0) {
    for (const auto& e : vr.errors) {
      std::cerr << "  " << e << "\n";
    }
  }
}

pargraph::MpiVertexPart must_partition(const pargraph::Graph& g) {
  const pargraph::MpiVertexPart part = pargraph::mpi_partition_vertices_1d(g);
  std::uint64_t local_e = part.local.num_edges;
  std::uint64_t sum_e = 0;
  mpi_check(MPI_Allreduce(&local_e, &sum_e, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce edges");
  expect(sum_e == g.num_edges, "1D slices cover every edge");

  int bounds[2] = {static_cast<int>(part.lo), static_cast<int>(part.hi)};
  std::vector<int> all(static_cast<std::size_t>(g_ranks) * 2);
  mpi_check(MPI_Allgather(bounds, 2, MPI_INT, all.data(), 2, MPI_INT,
                          MPI_COMM_WORLD),
            "MPI_Allgather bounds");
  bool tiled = all[0] == 0 &&
               all[all.size() - 1] == static_cast<int>(g.num_vertices);
  for (int r = 0; r + 1 < g_ranks; ++r) {
    if (all[static_cast<std::size_t>(2 * r + 1)] !=
        all[static_cast<std::size_t>(2 * (r + 1))]) {
      tiled = false;
    }
    if (all[static_cast<std::size_t>(2 * r)] >
        all[static_cast<std::size_t>(2 * r + 1)]) {
      tiled = false;
    }
  }
  expect(tiled, "1D ranges tile [0, V)");
  expect(part.local.num_vertices ==
             static_cast<pargraph::offset_t>(part.hi - part.lo),
         "local vertex count matches ownership range");
  return part;
}

}  // namespace

int run_bfs_mpi_atomic_tests() {
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank), "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &g_ranks), "MPI_Comm_size");
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int rc = 0;
  try {
    const fs::path el = find_tiny("bfs_sample.el");
    expect(!el.empty(), "locate bfs_sample.el");
    if (el.empty()) {
      throw std::runtime_error("missing bfs_sample.el");
    }

    const fs::path out_dir =
        fs::path(PARGRAPH_SOURCE_DIR) / "tests" / "tiny_graphs";
    const fs::path csr_path = out_dir / "bfs_sample.csr";
    const fs::path results_dir = fs::path(PARGRAPH_SOURCE_DIR) / "results";
    if (g_rank == 0) {
      fs::create_directories(results_dir);
    }
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier results dir");

    pargraph::BuildOptions opts;
    opts.make_undirected = true;
    opts.dedupe = true;
    opts.drop_self_loops = true;
    opts.sort_adjacency = true;

    pargraph::Graph built = pargraph::load_edgelist_text(el.string(), opts);
    expect(built.num_vertices == 6, "sample V == 6");
    expect(built.num_edges == 12, "sample E == 12 (symmetrized)");
    expect(pargraph::validate_csr(built).ok, "validate sample CSR");
    if (g_rank == 0) {
      pargraph::save_csr_binary(built, csr_path.string());
    }
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier csr save");
    pargraph::Graph g = pargraph::load_csr_binary(csr_path.string());
    pargraph::MpiVertexPart part = must_partition(g);

    const int thread_counts[] = {1, 2};
    for (int nt : thread_counts) {
      check_against_serial(g, part, 0, nt, "sample BFS(0)");
      check_against_serial(g, part, 3, nt, "sample BFS(3)");
    }

    {
      const pargraph::BfsResult serial = pargraph::bfs_serial(g, 0);
      expect(serial.dist[0] == 0 && serial.dist[1] == 1 && serial.dist[2] == 2 &&
                 serial.dist[3] == 3 && serial.dist[4] == 1 &&
                 serial.dist[5] == 2,
             "serial BFS(0) hand distances");
      const pargraph::BfsResult serial3 = pargraph::bfs_serial(g, 3);
      expect(serial3.dist[3] == 0 && serial3.dist[2] == 1 &&
                 serial3.dist[1] == 2 && serial3.dist[5] == 2 &&
                 serial3.dist[0] == 3 && serial3.dist[4] == 3,
             "serial BFS(3) hand distances");
    }

    {
      const pargraph::BfsResult once = pargraph::bfs_mpi_atomic(part, 0, 2);
      const pargraph::BfsResult twice = pargraph::bfs_mpi_atomic(part, 0, 2);
      expect(once.dist == twice.dist && once.visited == twice.visited,
             "repeated mpi_atomic BFS(0) distances agree");
      if (g_ranks == 1) {
        expect(once.mpi.bytes_sent == 0, "1 rank sends no discovery payload");
      } else {
        expect(once.mpi.bytes_sent > 0, "multi-rank sample sends discovery payload");
      }
    }

    // Directed chain forces a cross-rank edge whenever more than one rank
    // owns a vertex. Distances are 0,1,2,3,4,5 from source 0.
    {
      std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
          {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
      pargraph::BuildOptions dopts;
      dopts.make_undirected = false;
      dopts.dedupe = true;
      dopts.drop_self_loops = true;
      dopts.sort_adjacency = true;
      pargraph::Graph chain =
          pargraph::build_csr_from_edges(std::move(edges), 6, dopts);
      pargraph::MpiVertexPart chain_part = must_partition(chain);
      check_against_serial(chain, chain_part, 0, 2, "directed chain");
      const pargraph::BfsResult mpi_r =
          pargraph::bfs_mpi_atomic(chain_part, 0, 1);
      if (g_ranks == 1) {
        expect(mpi_r.mpi.bytes_sent == 0, "chain 1 rank bytes_sent == 0");
      } else {
        expect(mpi_r.mpi.bytes_sent > 0, "chain crosses ranks");
      }
    }

    // Two predecessors at the same level. Parent may differ from serial;
    // distances and the Graph500 parent checks must still hold.
    {
      std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
          {0, 1}, {0, 2}, {1, 3}, {2, 3}};
      pargraph::BuildOptions dopts;
      dopts.make_undirected = false;
      dopts.sort_adjacency = true;
      pargraph::Graph diamond =
          pargraph::build_csr_from_edges(std::move(edges), 4, dopts);
      pargraph::MpiVertexPart diamond_part = must_partition(diamond);
      check_against_serial(diamond, diamond_part, 0, 2, "diamond");
      const pargraph::BfsResult mpi_r =
          pargraph::bfs_mpi_atomic(diamond_part, 0, 2);
      const bool parent_ok = mpi_r.parent[3] == 1 || mpi_r.parent[3] == 2;
      expect(parent_ok, "diamond parent of 3 is 1 or 2");
    }

    {
      std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {{0, 1},
                                                                        {2, 3}};
      pargraph::BuildOptions dopts;
      dopts.make_undirected = false;
      dopts.sort_adjacency = true;
      pargraph::Graph split =
          pargraph::build_csr_from_edges(std::move(edges), 4, dopts);
      pargraph::MpiVertexPart split_part = must_partition(split);
      check_against_serial(split, split_part, 0, 2, "disconnected");
      const pargraph::BfsResult mpi_r =
          pargraph::bfs_mpi_atomic(split_part, 0, 1);
      expect(mpi_r.dist[2] == pargraph::kBfsUnreached &&
                 mpi_r.dist[3] == pargraph::kBfsUnreached &&
                 mpi_r.parent[2] == -1 && mpi_r.parent[3] == -1,
             "disconnected vertices stay unreached");
    }

    {
      pargraph::Graph iso;
      iso.num_vertices = 3;
      iso.num_edges = 0;
      iso.offsets = {0, 0, 0, 0};
      iso.directed = true;
      iso.sorted = true;
      pargraph::MpiVertexPart iso_part = must_partition(iso);
      check_against_serial(iso, iso_part, 1, 2, "isolated");
      const pargraph::BfsResult mpi_r = pargraph::bfs_mpi_atomic(iso_part, 1, 1);
      expect(mpi_r.visited == 1 && mpi_r.edges_examined == 0 &&
                 mpi_r.dist[1] == 0 && mpi_r.parent[1] == 1,
             "isolated vertex counts");
    }

    {
      pargraph::Graph empty;
      empty.offsets = {0};
      empty.sorted = true;
      pargraph::MpiVertexPart empty_part = must_partition(empty);
      const pargraph::BfsResult mpi_r =
          pargraph::bfs_mpi_atomic(empty_part, 0, 1);
      expect(mpi_r.visited == 0 && mpi_r.dist.empty(), "empty graph");
      const pargraph::BfsResult bad = pargraph::bfs_mpi_atomic(part, 999, 1);
      expect(bad.visited == 0 &&
                 bad.dist.size() == static_cast<std::size_t>(g.num_vertices),
             "out-of-range source");
    }

    {
      const int nt = 2;
      const fs::path results_csv =
          results_dir / ("bfs_sample_mpi_atomic_np" + std::to_string(g_ranks) +
                         ".csv");
      pargraph::RunMetadata meta = pargraph::make_run_metadata();
      meta.kernel = "bfs";
      meta.variant = "mpi_atomic";
      meta.graph_path = csr_path.string();
      meta.num_vertices = g.num_vertices;
      meta.num_edges = g.num_edges;
      meta.directed = g.directed;
      meta.seed = 42;
      meta.sources = 2;
      meta.trials = 2;
      meta.threads = nt;
      meta.ranks = g_ranks;
      meta.load_time_ms = 0.0;

      std::vector<pargraph::BfsTrialRow> rows;
      std::vector<double> timed_ms;
      std::vector<double> timed_teps;
      std::vector<double> timed_compute;
      std::vector<double> timed_comm;
      std::uint64_t bytes_sum = 0;
      const pargraph::vtx_t srcs[] = {0, 3};

      for (pargraph::vtx_t src : srcs) {
        const pargraph::BfsResult serial = pargraph::bfs_serial(g, src);
        for (int pass = -1; pass < meta.trials; ++pass) {
          mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier trial");
          pargraph::Timer t;
          t.start();
          pargraph::BfsResult result = pargraph::bfs_mpi_atomic(part, src, nt);
          t.stop();
          const bool ok = pargraph::validate_bfs(g, src, result).ok &&
                          distances_match(result, serial);
          expect(ok, "per-trial mpi_atomic validate source=" +
                         std::to_string(src) + " pass=" + std::to_string(pass));
          if (g_rank != 0) {
            continue;
          }
          pargraph::BfsTrialRow row;
          row.source = static_cast<int>(src);
          row.trial = pass;
          row.is_warmup = (pass < 0);
          row.time_ms = t.elapsed_ms();
          row.visited = result.visited;
          row.edges_examined = result.edges_examined;
          row.teps = (t.elapsed_sec() > 0.0)
                         ? static_cast<double>(result.edges_examined) /
                               t.elapsed_sec()
                         : 0.0;
          row.validated = true;
          row.valid_ok = ok;
          row.compute_ms = result.mpi.compute_sec * 1e3;
          row.comm_ms = result.mpi.comm_sec * 1e3;
          row.bytes_sent = result.mpi.bytes_sent;
          rows.push_back(row);
          if (!row.is_warmup) {
            timed_ms.push_back(row.time_ms);
            timed_teps.push_back(row.teps);
            timed_compute.push_back(row.compute_ms);
            timed_comm.push_back(row.comm_ms);
            bytes_sum += row.bytes_sent;
          }
        }
      }

      if (g_rank == 0) {
        pargraph::write_bfs_results_csv(results_csv.string(), meta, rows);
        const double med = pargraph::median_sorted(timed_ms);
        const double hteps = pargraph::harmonic_mean(timed_teps);
        const double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
        const double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
        pargraph::write_bfs_results_meta_json(
            results_csv.string() + ".meta.json", meta, med, min_ms, max_ms,
            hteps, timed_ms.size(), pargraph::median_sorted(timed_compute),
            pargraph::median_sorted(timed_comm), bytes_sum);
        expect(fs::exists(results_csv), "wrote mpi_atomic sample results CSV");
        expect(fs::exists(results_csv.string() + ".meta.json"),
               "wrote mpi_atomic sample results meta JSON");
      }
    }
  } catch (const std::exception& ex) {
    if (g_rank == 0) {
      std::cerr << "FAIL: exception: " << ex.what() << "\n";
    }
    ++g_failures;
  }

  int global_failures = 0;
  MPI_Allreduce(&g_failures, &global_failures, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (g_rank == 0) {
    if (global_failures == 0) {
      std::cout << "\nAll MPI atomic BFS tests passed (" << g_ranks
                << " ranks).\n";
    } else {
      std::cerr << "\n" << global_failures << " assertion(s) failed.\n";
    }
  }
  return global_failures == 0 ? 0 : 1;
}

#ifndef PARGRAPH_EMBED_MPI_TEST
int main() {
  int provided = 0;
  mpi_check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided),
            "MPI_Init_thread");
  const int rc = run_bfs_mpi_atomic_tests();
  MPI_Finalize();
  return rc;
}
#endif
