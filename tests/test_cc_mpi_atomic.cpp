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

pargraph::MpiVertexPart must_partition(const pargraph::Graph& g) {
  pargraph::MpiVertexPart part = pargraph::mpi_partition_vertices_1d(g);
  expect(part.ranks == g_ranks, "partition ranks match world");
  expect(part.lo <= part.hi, "ownership range ordered");
  expect(part.hi <= g.num_vertices, "ownership within V");
  return part;
}

bool matches_serial_local(const pargraph::Graph& g,
                          const pargraph::CcResult& r) {
  const pargraph::CcResult serial = pargraph::cc_serial_unionfind(g);
  return r.num_components == serial.num_components &&
         r.largest_component == serial.largest_component &&
         pargraph::cc_same_partition(r.label, serial.label) &&
         pargraph::validate_cc(g, r).ok;
}

void check_against_serial(const pargraph::Graph& g,
                          const pargraph::MpiVertexPart& part, int nt,
                          const std::string& tag) {
  const pargraph::CcResult serial = pargraph::cc_serial_unionfind(g);
  const pargraph::CcResult mpi_r = pargraph::cc_mpi_atomic(part, nt);
  expect(pargraph::validate_cc(g, mpi_r).ok, tag + " partition validate");
  expect(mpi_r.num_components == serial.num_components &&
             mpi_r.largest_component == serial.largest_component &&
             pargraph::cc_same_partition(mpi_r.label, serial.label),
         tag + " matches serial partition threads=" + std::to_string(nt));
}

}  // namespace

int run_cc_mpi_atomic_tests() {
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank), "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &g_ranks), "MPI_Comm_size");
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  try {
    const fs::path el = find_tiny("cc_sample.el");
    expect(!el.empty(), "locate cc_sample.el");
    if (el.empty()) {
      throw std::runtime_error("missing cc_sample.el");
    }

    const fs::path out_dir =
        fs::path(PARGRAPH_SOURCE_DIR) / "tests" / "tiny_graphs";
    const fs::path csr_path = out_dir / "cc_sample.csr";
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
    expect(built.num_edges == 10, "sample E == 10");
    if (g_rank == 0) {
      pargraph::save_csr_binary(built, csr_path.string());
    }
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier csr save");
    pargraph::Graph g = pargraph::load_csr_binary(csr_path.string());
    pargraph::MpiVertexPart part = must_partition(g);

    for (int nt : {1, 2}) {
      check_against_serial(g, part, nt, "sample CC");
    }

    {
      std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
          {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
      pargraph::BuildOptions dopts;
      dopts.make_undirected = false;
      dopts.dedupe = true;
      dopts.sort_adjacency = true;
      pargraph::Graph chain =
          pargraph::build_csr_from_edges(std::move(edges), 6, dopts);
      pargraph::MpiVertexPart chain_part = must_partition(chain);
      check_against_serial(chain, chain_part, 2, "directed chain WCC");
    }

    {
      std::vector<std::pair<pargraph::vtx_t, pargraph::vtx_t>> edges = {
          {0, 1}, {2, 3}};
      pargraph::BuildOptions dopts;
      dopts.make_undirected = true;
      dopts.sort_adjacency = true;
      pargraph::Graph split =
          pargraph::build_csr_from_edges(std::move(edges), 4, dopts);
      pargraph::MpiVertexPart split_part = must_partition(split);
      check_against_serial(split, split_part, 2, "two components");
      const pargraph::CcResult r = pargraph::cc_mpi_atomic(split_part, 1);
      expect(r.num_components == 2, "split has 2 components");
    }

    {
      pargraph::Graph empty;
      empty.offsets = {0};
      empty.sorted = true;
      pargraph::MpiVertexPart empty_part = must_partition(empty);
      const pargraph::CcResult r = pargraph::cc_mpi_atomic(empty_part, 1);
      expect(r.num_components == 0 && r.label.empty(), "empty graph");
    }

    {
      const int nt = 2;
      const fs::path results_csv =
          results_dir / ("cc_sample_mpi_atomic_np" + std::to_string(g_ranks) +
                         ".csv");
      pargraph::RunMetadata meta = pargraph::make_run_metadata();
      meta.kernel = "cc";
      meta.variant = "mpi_atomic";
      meta.algo = "labelprop";
      meta.graph_path = csr_path.string();
      meta.num_vertices = g.num_vertices;
      meta.num_edges = g.num_edges;
      meta.directed = g.directed;
      meta.trials = 2;
      meta.threads = nt;
      meta.ranks = g_ranks;

      std::vector<pargraph::CcTrialRow> rows;
      std::vector<double> timed_ms;
      std::vector<double> timed_compute;
      std::vector<double> timed_comm;
      std::uint64_t bytes_sum = 0;
      std::uint64_t comps = 0;
      std::uint64_t largest = 0;
      std::uint64_t rounds = 0;

      for (int pass = -1; pass < meta.trials; ++pass) {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier trial");
        pargraph::Timer t;
        t.start();
        pargraph::CcResult r = pargraph::cc_mpi_atomic(part, nt);
        t.stop();
        if (g_rank == 0) {
          pargraph::CcTrialRow row;
          row.trial = pass;
          row.is_warmup = (pass < 0);
          row.time_ms = t.elapsed_ms();
          row.num_components = r.num_components;
          row.largest_component = r.largest_component;
          row.edges_united = r.edges_united;
          row.rounds = r.rounds;
          row.compute_ms = r.mpi.compute_sec * 1e3;
          row.comm_ms = r.mpi.comm_sec * 1e3;
          row.bytes_sent = r.mpi.bytes_sent;
          row.validated = true;
          row.valid_ok = matches_serial_local(g, r);
          rows.push_back(row);
          comps = r.num_components;
          largest = r.largest_component;
          rounds = r.rounds;
          if (!row.is_warmup) {
            timed_ms.push_back(row.time_ms);
            timed_compute.push_back(row.compute_ms);
            timed_comm.push_back(row.comm_ms);
            bytes_sum += row.bytes_sent;
          }
        }
      }

      if (g_rank == 0) {
        pargraph::write_cc_results_csv(results_csv.string(), meta, rows);
        const double med = pargraph::median_sorted(timed_ms);
        double min_ms = *std::min_element(timed_ms.begin(), timed_ms.end());
        double max_ms = *std::max_element(timed_ms.begin(), timed_ms.end());
        const double med_compute = pargraph::median_sorted(timed_compute);
        const double med_comm = pargraph::median_sorted(timed_comm);
        pargraph::write_cc_results_meta_json(
            results_csv.string() + ".meta.json", meta, med, min_ms, max_ms,
            comps, largest, timed_ms.size(), rounds, med_compute, med_comm,
            bytes_sum);
        expect(fs::exists(results_csv), "wrote mpi sample CSV");
      }
    }
  } catch (const std::exception& ex) {
    if (g_rank == 0) {
      std::cerr << "error: " << ex.what() << "\n";
    }
    ++g_failures;
  }

  int global_failures = 0;
  MPI_Allreduce(&g_failures, &global_failures, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (g_rank == 0) {
    if (global_failures == 0) {
      std::cout << "\nAll MPI atomic CC tests passed (" << g_ranks
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
  const int rc = run_cc_mpi_atomic_tests();
  MPI_Finalize();
  return rc;
}
#endif
