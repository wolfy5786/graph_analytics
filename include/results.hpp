#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pargraph {

struct RunMetadata {
  std::string timestamp_utc;
  std::string git_commit;
  std::string hostname;
  std::string os;
  std::string cpu;
  std::string compiler;
  std::string cmake_build_type;
  std::string kernel;
  std::string variant;
  std::string graph_path;
  std::uint64_t num_vertices = 0;
  std::uint64_t num_edges = 0;
  bool directed = true;
  int threads = 1;
  int ranks = 1;
  std::uint64_t seed = 0;
  int sources = 0;
  int trials = 0;
  double load_time_ms = 0.0;  // preprocessing, separate from kernel
  double distribution_time_ms = 0.0;  // MPI 1D slice, separate from kernel
  double orient_time_ms = 0.0;  // TC orientation, separate from kernel
  std::string algo;           // e.g. unionfind for CC
  std::string orient;         // TC: degree|id
  std::string intersect;      // TC: merge|galloping
  // PageRank parameters (ignored by other kernels).
  double damping = 0.85;
  double tolerance = 1e-6;
  int max_iters = 100;
};

inline bool reports_mpi_split(const RunMetadata& meta) {
  return meta.ranks > 1 || meta.variant.rfind("mpi", 0) == 0 ||
         meta.distribution_time_ms > 0.0;
}

struct BfsTrialRow {
  int source = 0;
  int trial = 0;  // -1 = warmup
  bool is_warmup = false;
  double time_ms = 0.0;
  std::uint64_t visited = 0;
  std::uint64_t edges_examined = 0;
  double teps = 0.0;
  bool validated = false;
  bool valid_ok = false;
  double compute_ms = 0.0;
  double comm_ms = 0.0;
  std::uint64_t bytes_sent = 0;
};

inline std::string csv_escape(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) {
    return s;
  }
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

inline void write_bfs_results_csv(const std::string& path,
                                  const RunMetadata& meta,
                                  const std::vector<BfsTrialRow>& rows) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results CSV: " + path);
  }

  const bool mpi_cols = reports_mpi_split(meta);
  out << "timestamp_utc,git_commit,hostname,os,cpu,compiler,build_type,"
         "kernel,variant,graph,num_vertices,num_edges,directed,"
         "threads,ranks,seed,sources_requested,trials_requested,load_time_ms,"
         "source,trial,is_warmup,time_ms,visited,edges_examined,teps,"
         "validated,valid_ok";
  if (mpi_cols) {
    out << ",compute_ms,comm_ms,bytes_sent";
  }
  out << "\n";

  for (const auto& r : rows) {
    out << csv_escape(meta.timestamp_utc) << ','
        << csv_escape(meta.git_commit) << ',' << csv_escape(meta.hostname)
        << ',' << csv_escape(meta.os) << ',' << csv_escape(meta.cpu) << ','
        << csv_escape(meta.compiler) << ','
        << csv_escape(meta.cmake_build_type) << ','
        << csv_escape(meta.kernel) << ',' << csv_escape(meta.variant) << ','
        << csv_escape(meta.graph_path) << ',' << meta.num_vertices << ','
        << meta.num_edges << ',' << (meta.directed ? 1 : 0) << ','
        << meta.threads << ',' << meta.ranks << ',' << meta.seed << ','
        << meta.sources << ',' << meta.trials << ',' << meta.load_time_ms
        << ',' << r.source << ',' << r.trial << ',' << (r.is_warmup ? 1 : 0)
        << ',' << r.time_ms << ',' << r.visited << ',' << r.edges_examined
        << ',' << r.teps << ',' << (r.validated ? 1 : 0) << ','
        << (r.valid_ok ? 1 : 0);
    if (mpi_cols) {
      out << ',' << r.compute_ms << ',' << r.comm_ms << ',' << r.bytes_sent;
    }
    out << '\n';
  }
}

inline void write_bfs_results_meta_json(const std::string& path,
                                        const RunMetadata& meta,
                                        double median_time_ms,
                                        double min_time_ms,
                                        double max_time_ms,
                                        double harmonic_mean_teps,
                                        std::uint64_t timed_trials,
                                        double median_compute_ms = 0.0,
                                        double median_comm_ms = 0.0,
                                        std::uint64_t bytes_sent = 0) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results meta JSON: " + path);
  }

  auto q = [](const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
      if (c == '"' || c == '\\') {
        o << '\\';
      }
      o << c;
    }
    o << '"';
    return o.str();
  };

  const bool mpi_stats = reports_mpi_split(meta);
  std::string notes =
      "Kernel time excludes graph load/preprocess. "
      "TEPS = edges_examined / time_sec (Graph500-style).";
  if (mpi_stats) {
    notes +=
        " distribution_time_ms is the 1D edge slice and is excluded from "
        "kernel time. compute_ms/comm_ms are the slowest rank's phase split "
        "(comm includes wait inside MPI). bytes_sent sums discovery payloads "
        "across ranks.";
  }

  out << "{\n"
      << "  \"timestamp_utc\": " << q(meta.timestamp_utc) << ",\n"
      << "  \"git_commit\": " << q(meta.git_commit) << ",\n"
      << "  \"hostname\": " << q(meta.hostname) << ",\n"
      << "  \"os\": " << q(meta.os) << ",\n"
      << "  \"cpu\": " << q(meta.cpu) << ",\n"
      << "  \"compiler\": " << q(meta.compiler) << ",\n"
      << "  \"build_type\": " << q(meta.cmake_build_type) << ",\n"
      << "  \"kernel\": " << q(meta.kernel) << ",\n"
      << "  \"variant\": " << q(meta.variant) << ",\n"
      << "  \"graph\": " << q(meta.graph_path) << ",\n"
      << "  \"num_vertices\": " << meta.num_vertices << ",\n"
      << "  \"num_edges\": " << meta.num_edges << ",\n"
      << "  \"directed\": " << (meta.directed ? "true" : "false") << ",\n"
      << "  \"threads\": " << meta.threads << ",\n"
      << "  \"ranks\": " << meta.ranks << ",\n"
      << "  \"seed\": " << meta.seed << ",\n"
      << "  \"sources\": " << meta.sources << ",\n"
      << "  \"trials\": " << meta.trials << ",\n"
      << "  \"load_time_ms\": " << meta.load_time_ms << ",\n";
  if (mpi_stats) {
    const double phase = median_compute_ms + median_comm_ms;
    const double share = (phase > 0.0) ? (median_comm_ms / phase) : 0.0;
    out << "  \"distribution_time_ms\": " << meta.distribution_time_ms << ",\n"
        << "  \"median_compute_ms\": " << median_compute_ms << ",\n"
        << "  \"median_comm_ms\": " << median_comm_ms << ",\n"
        << "  \"comm_share_of_phase\": " << share << ",\n"
        << "  \"bytes_sent\": " << bytes_sent << ",\n";
  }
  out << "  \"timed_trials\": " << timed_trials << ",\n"
      << "  \"median_kernel_time_ms\": " << median_time_ms << ",\n"
      << "  \"min_kernel_time_ms\": " << min_time_ms << ",\n"
      << "  \"max_kernel_time_ms\": " << max_time_ms << ",\n"
      << "  \"harmonic_mean_teps\": " << harmonic_mean_teps << ",\n"
      << "  \"notes\": " << q(notes) << "\n"
      << "}\n";
}

struct CcTrialRow {
  int trial = 0;  // -1 = warmup
  bool is_warmup = false;
  double time_ms = 0.0;
  std::uint64_t num_components = 0;
  std::uint64_t largest_component = 0;
  std::uint64_t edges_united = 0;
  std::uint64_t rounds = 0;
  bool validated = false;
  bool valid_ok = false;
  double compute_ms = 0.0;
  double comm_ms = 0.0;
  std::uint64_t bytes_sent = 0;
};

inline void write_cc_results_csv(const std::string& path,
                                 const RunMetadata& meta,
                                 const std::vector<CcTrialRow>& rows) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results CSV: " + path);
  }

  const bool mpi_cols = reports_mpi_split(meta);
  out << "timestamp_utc,git_commit,hostname,os,cpu,compiler,build_type,"
         "kernel,variant,algo,graph,num_vertices,num_edges,directed,"
         "threads,ranks,seed,trials_requested,load_time_ms,"
         "trial,is_warmup,time_ms,num_components,largest_component,"
         "edges_united,rounds,validated,valid_ok";
  if (mpi_cols) {
    out << ",compute_ms,comm_ms,bytes_sent";
  }
  out << "\n";

  for (const auto& r : rows) {
    out << csv_escape(meta.timestamp_utc) << ','
        << csv_escape(meta.git_commit) << ',' << csv_escape(meta.hostname)
        << ',' << csv_escape(meta.os) << ',' << csv_escape(meta.cpu) << ','
        << csv_escape(meta.compiler) << ','
        << csv_escape(meta.cmake_build_type) << ','
        << csv_escape(meta.kernel) << ',' << csv_escape(meta.variant) << ','
        << csv_escape(meta.algo) << ',' << csv_escape(meta.graph_path) << ','
        << meta.num_vertices << ',' << meta.num_edges << ','
        << (meta.directed ? 1 : 0) << ',' << meta.threads << ',' << meta.ranks
        << ',' << meta.seed << ',' << meta.trials << ',' << meta.load_time_ms
        << ',' << r.trial << ',' << (r.is_warmup ? 1 : 0) << ',' << r.time_ms
        << ',' << r.num_components << ',' << r.largest_component << ','
        << r.edges_united << ',' << r.rounds << ',' << (r.validated ? 1 : 0)
        << ',' << (r.valid_ok ? 1 : 0);
    if (mpi_cols) {
      out << ',' << r.compute_ms << ',' << r.comm_ms << ',' << r.bytes_sent;
    }
    out << '\n';
  }
}

inline void write_cc_results_meta_json(const std::string& path,
                                       const RunMetadata& meta,
                                       double median_time_ms,
                                       double min_time_ms,
                                       double max_time_ms,
                                       std::uint64_t num_components,
                                       std::uint64_t largest_component,
                                       std::uint64_t timed_trials,
                                       std::uint64_t rounds = 0,
                                       double median_compute_ms = 0.0,
                                       double median_comm_ms = 0.0,
                                       std::uint64_t bytes_sent = 0) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results meta JSON: " + path);
  }

  auto q = [](const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
      if (c == '"' || c == '\\') {
        o << '\\';
      }
      o << c;
    }
    o << '"';
    return o.str();
  };

  const bool mpi_stats = reports_mpi_split(meta);
  std::string notes =
      "Kernel time excludes graph load/preprocess. "
      "On directed CSR, each arc is treated as undirected (WCC). "
      "Parallel variants compare partitions to serial union-find "
      "(label values may differ). rounds counts hooking / LP iterations.";
  if (mpi_stats) {
    notes +=
        " distribution_time_ms is the 1D edge slice and is excluded from "
        "kernel time. compute_ms/comm_ms are the slowest rank's phase split "
        "(comm includes wait inside MPI). bytes_sent sums Allreduce payloads.";
  }

  out << "{\n"
      << "  \"timestamp_utc\": " << q(meta.timestamp_utc) << ",\n"
      << "  \"git_commit\": " << q(meta.git_commit) << ",\n"
      << "  \"hostname\": " << q(meta.hostname) << ",\n"
      << "  \"os\": " << q(meta.os) << ",\n"
      << "  \"cpu\": " << q(meta.cpu) << ",\n"
      << "  \"compiler\": " << q(meta.compiler) << ",\n"
      << "  \"build_type\": " << q(meta.cmake_build_type) << ",\n"
      << "  \"kernel\": " << q(meta.kernel) << ",\n"
      << "  \"variant\": " << q(meta.variant) << ",\n"
      << "  \"algo\": " << q(meta.algo) << ",\n"
      << "  \"graph\": " << q(meta.graph_path) << ",\n"
      << "  \"num_vertices\": " << meta.num_vertices << ",\n"
      << "  \"num_edges\": " << meta.num_edges << ",\n"
      << "  \"directed\": " << (meta.directed ? "true" : "false") << ",\n"
      << "  \"threads\": " << meta.threads << ",\n"
      << "  \"ranks\": " << meta.ranks << ",\n"
      << "  \"seed\": " << meta.seed << ",\n"
      << "  \"trials\": " << meta.trials << ",\n"
      << "  \"load_time_ms\": " << meta.load_time_ms << ",\n";
  if (mpi_stats) {
    const double phase = median_compute_ms + median_comm_ms;
    const double share = (phase > 0.0) ? (median_comm_ms / phase) : 0.0;
    out << "  \"distribution_time_ms\": " << meta.distribution_time_ms << ",\n"
        << "  \"median_compute_ms\": " << median_compute_ms << ",\n"
        << "  \"median_comm_ms\": " << median_comm_ms << ",\n"
        << "  \"comm_share_of_phase\": " << share << ",\n"
        << "  \"bytes_sent\": " << bytes_sent << ",\n";
  }
  out << "  \"timed_trials\": " << timed_trials << ",\n"
      << "  \"median_kernel_time_ms\": " << median_time_ms << ",\n"
      << "  \"min_kernel_time_ms\": " << min_time_ms << ",\n"
      << "  \"max_kernel_time_ms\": " << max_time_ms << ",\n"
      << "  \"num_components\": " << num_components << ",\n"
      << "  \"largest_component\": " << largest_component << ",\n"
      << "  \"rounds\": " << rounds << ",\n"
      << "  \"notes\": " << q(notes) << "\n"
      << "}\n";
}

struct TcTrialRow {
  int trial = 0;  // -1 = warmup
  bool is_warmup = false;
  double time_ms = 0.0;
  std::uint64_t triangles = 0;
  std::uint64_t wedges_examined = 0;
  std::uint64_t intersect_ops = 0;
  double intersect_rate = 0.0;  // intersect_ops / time_sec
  bool validated = false;
  bool valid_ok = false;
};

inline void write_tc_results_csv(const std::string& path,
                                 const RunMetadata& meta,
                                 const std::vector<TcTrialRow>& rows) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results CSV: " + path);
  }

  out << "timestamp_utc,git_commit,hostname,os,cpu,compiler,build_type,"
         "kernel,variant,orient,intersect,graph,num_vertices,num_edges,"
         "directed,threads,ranks,seed,trials_requested,load_time_ms,"
         "orient_time_ms,trial,is_warmup,time_ms,triangles,wedges_examined,"
         "intersect_ops,intersect_rate,validated,valid_ok\n";

  for (const auto& r : rows) {
    out << csv_escape(meta.timestamp_utc) << ','
        << csv_escape(meta.git_commit) << ',' << csv_escape(meta.hostname)
        << ',' << csv_escape(meta.os) << ',' << csv_escape(meta.cpu) << ','
        << csv_escape(meta.compiler) << ','
        << csv_escape(meta.cmake_build_type) << ','
        << csv_escape(meta.kernel) << ',' << csv_escape(meta.variant) << ','
        << csv_escape(meta.orient) << ',' << csv_escape(meta.intersect) << ','
        << csv_escape(meta.graph_path) << ',' << meta.num_vertices << ','
        << meta.num_edges << ',' << (meta.directed ? 1 : 0) << ','
        << meta.threads << ',' << meta.ranks << ',' << meta.seed << ','
        << meta.trials << ',' << meta.load_time_ms << ','
        << meta.orient_time_ms << ',' << r.trial << ','
        << (r.is_warmup ? 1 : 0) << ',' << r.time_ms << ',' << r.triangles
        << ',' << r.wedges_examined << ',' << r.intersect_ops << ','
        << r.intersect_rate << ',' << (r.validated ? 1 : 0) << ','
        << (r.valid_ok ? 1 : 0) << '\n';
  }
}

inline void write_tc_results_meta_json(const std::string& path,
                                       const RunMetadata& meta,
                                       double median_time_ms,
                                       double min_time_ms,
                                       double max_time_ms,
                                       std::uint64_t triangles,
                                       double median_intersect_rate,
                                       std::uint64_t timed_trials) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results meta JSON: " + path);
  }

  auto q = [](const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
      if (c == '"' || c == '\\') {
        o << '\\';
      }
      o << c;
    }
    o << '"';
    return o.str();
  };

  out << "{\n"
      << "  \"timestamp_utc\": " << q(meta.timestamp_utc) << ",\n"
      << "  \"git_commit\": " << q(meta.git_commit) << ",\n"
      << "  \"hostname\": " << q(meta.hostname) << ",\n"
      << "  \"os\": " << q(meta.os) << ",\n"
      << "  \"cpu\": " << q(meta.cpu) << ",\n"
      << "  \"compiler\": " << q(meta.compiler) << ",\n"
      << "  \"build_type\": " << q(meta.cmake_build_type) << ",\n"
      << "  \"kernel\": " << q(meta.kernel) << ",\n"
      << "  \"variant\": " << q(meta.variant) << ",\n"
      << "  \"orient\": " << q(meta.orient) << ",\n"
      << "  \"intersect\": " << q(meta.intersect) << ",\n"
      << "  \"graph\": " << q(meta.graph_path) << ",\n"
      << "  \"num_vertices\": " << meta.num_vertices << ",\n"
      << "  \"num_edges\": " << meta.num_edges << ",\n"
      << "  \"directed\": " << (meta.directed ? "true" : "false") << ",\n"
      << "  \"threads\": " << meta.threads << ",\n"
      << "  \"ranks\": " << meta.ranks << ",\n"
      << "  \"seed\": " << meta.seed << ",\n"
      << "  \"trials\": " << meta.trials << ",\n"
      << "  \"load_time_ms\": " << meta.load_time_ms << ",\n"
      << "  \"orient_time_ms\": " << meta.orient_time_ms << ",\n"
      << "  \"timed_trials\": " << timed_trials << ",\n"
      << "  \"median_kernel_time_ms\": " << median_time_ms << ",\n"
      << "  \"min_kernel_time_ms\": " << min_time_ms << ",\n"
      << "  \"max_kernel_time_ms\": " << max_time_ms << ",\n"
      << "  \"triangles\": " << triangles << ",\n"
      << "  \"median_intersect_rate\": " << median_intersect_rate << ",\n"
      << "  \"notes\": \"Kernel time excludes graph load and orientation. "
         "intersect_rate = intersect_ops / time_sec. Forward algorithm: "
         "orient edges by degree/id, intersect sorted forward adjacency.\"\n"
      << "}\n";
}

struct PageRankTrialRow {
  int trial = 0;  // -1 = warmup
  bool is_warmup = false;
  double time_ms = 0.0;
  int iterations = 0;
  double residual_l1 = 0.0;
  double score_sum = 0.0;
  std::uint64_t edges_examined = 0;
  double ms_per_iter = 0.0;
  bool validated = false;
  bool valid_ok = false;
  double compute_ms = 0.0;
  double comm_ms = 0.0;
  std::uint64_t bytes_sent = 0;
};

inline void write_pagerank_results_csv(const std::string& path,
                                       const RunMetadata& meta,
                                       const std::vector<PageRankTrialRow>& rows) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results CSV: " + path);
  }

  const bool mpi_cols = reports_mpi_split(meta);
  out << "timestamp_utc,git_commit,hostname,os,cpu,compiler,build_type,"
         "kernel,variant,graph,num_vertices,num_edges,directed,"
         "threads,ranks,seed,trials_requested,load_time_ms,"
         "damping,tolerance,max_iters,"
         "trial,is_warmup,time_ms,iterations,residual_l1,score_sum,"
         "edges_examined,ms_per_iter,validated,valid_ok";
  if (mpi_cols) {
    out << ",compute_ms,comm_ms,bytes_sent";
  }
  out << "\n";

  for (const auto& r : rows) {
    out << csv_escape(meta.timestamp_utc) << ','
        << csv_escape(meta.git_commit) << ',' << csv_escape(meta.hostname)
        << ',' << csv_escape(meta.os) << ',' << csv_escape(meta.cpu) << ','
        << csv_escape(meta.compiler) << ','
        << csv_escape(meta.cmake_build_type) << ','
        << csv_escape(meta.kernel) << ',' << csv_escape(meta.variant) << ','
        << csv_escape(meta.graph_path) << ',' << meta.num_vertices << ','
        << meta.num_edges << ',' << (meta.directed ? 1 : 0) << ','
        << meta.threads << ',' << meta.ranks << ',' << meta.seed << ','
        << meta.trials << ',' << meta.load_time_ms << ',' << meta.damping
        << ',' << meta.tolerance << ',' << meta.max_iters << ',' << r.trial
        << ',' << (r.is_warmup ? 1 : 0) << ',' << r.time_ms << ','
        << r.iterations << ',' << r.residual_l1 << ',' << r.score_sum << ','
        << r.edges_examined << ',' << r.ms_per_iter << ','
        << (r.validated ? 1 : 0) << ',' << (r.valid_ok ? 1 : 0);
    if (mpi_cols) {
      out << ',' << r.compute_ms << ',' << r.comm_ms << ',' << r.bytes_sent;
    }
    out << '\n';
  }
}

inline void write_pagerank_results_meta_json(
    const std::string& path, const RunMetadata& meta, double median_time_ms,
    double min_time_ms, double max_time_ms, double median_ms_per_iter,
    int last_iterations, double last_residual, std::uint64_t timed_trials,
    double median_compute_ms = 0.0, double median_comm_ms = 0.0,
    std::uint64_t bytes_sent = 0) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write results meta JSON: " + path);
  }

  auto q = [](const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
      if (c == '"' || c == '\\') {
        o << '\\';
      }
      o << c;
    }
    o << '"';
    return o.str();
  };

  const bool mpi_stats = reports_mpi_split(meta);
  std::string notes =
      "Kernel time excludes graph load/preprocess and reverse-CSR build "
      "when timed outside the kernel. PageRank reports total time to "
      "convergence and ms/iteration at the configured tolerance.";
  if (mpi_stats) {
    notes +=
        " distribution_time_ms is the 1D edge slice and is excluded from "
        "kernel time. compute_ms/comm_ms are the slowest rank's phase split "
        "(comm includes wait inside MPI). bytes_sent sums score/dangling "
        "payloads across ranks.";
  }

  out << "{\n"
      << "  \"timestamp_utc\": " << q(meta.timestamp_utc) << ",\n"
      << "  \"git_commit\": " << q(meta.git_commit) << ",\n"
      << "  \"hostname\": " << q(meta.hostname) << ",\n"
      << "  \"os\": " << q(meta.os) << ",\n"
      << "  \"cpu\": " << q(meta.cpu) << ",\n"
      << "  \"compiler\": " << q(meta.compiler) << ",\n"
      << "  \"build_type\": " << q(meta.cmake_build_type) << ",\n"
      << "  \"kernel\": " << q(meta.kernel) << ",\n"
      << "  \"variant\": " << q(meta.variant) << ",\n"
      << "  \"graph\": " << q(meta.graph_path) << ",\n"
      << "  \"num_vertices\": " << meta.num_vertices << ",\n"
      << "  \"num_edges\": " << meta.num_edges << ",\n"
      << "  \"directed\": " << (meta.directed ? "true" : "false") << ",\n"
      << "  \"threads\": " << meta.threads << ",\n"
      << "  \"ranks\": " << meta.ranks << ",\n"
      << "  \"seed\": " << meta.seed << ",\n"
      << "  \"trials\": " << meta.trials << ",\n"
      << "  \"load_time_ms\": " << meta.load_time_ms << ",\n"
      << "  \"damping\": " << meta.damping << ",\n"
      << "  \"tolerance\": " << meta.tolerance << ",\n"
      << "  \"max_iters\": " << meta.max_iters << ",\n";
  if (mpi_stats) {
    const double phase = median_compute_ms + median_comm_ms;
    const double share = (phase > 0.0) ? (median_comm_ms / phase) : 0.0;
    out << "  \"distribution_time_ms\": " << meta.distribution_time_ms << ",\n"
        << "  \"median_compute_ms\": " << median_compute_ms << ",\n"
        << "  \"median_comm_ms\": " << median_comm_ms << ",\n"
        << "  \"comm_share_of_phase\": " << share << ",\n"
        << "  \"bytes_sent\": " << bytes_sent << ",\n";
  }
  out << "  \"timed_trials\": " << timed_trials << ",\n"
      << "  \"median_kernel_time_ms\": " << median_time_ms << ",\n"
      << "  \"min_kernel_time_ms\": " << min_time_ms << ",\n"
      << "  \"max_kernel_time_ms\": " << max_time_ms << ",\n"
      << "  \"median_ms_per_iter\": " << median_ms_per_iter << ",\n"
      << "  \"last_iterations\": " << last_iterations << ",\n"
      << "  \"last_residual_l1\": " << last_residual << ",\n"
      << "  \"notes\": " << q(notes) << "\n"
      << "}\n";
}

}  // namespace pargraph
