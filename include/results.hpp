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
};

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

  out << "timestamp_utc,git_commit,hostname,os,cpu,compiler,build_type,"
         "kernel,variant,graph,num_vertices,num_edges,directed,"
         "threads,ranks,seed,sources_requested,trials_requested,load_time_ms,"
         "source,trial,is_warmup,time_ms,visited,edges_examined,teps,"
         "validated,valid_ok\n";

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
        << (r.valid_ok ? 1 : 0) << '\n';
  }
}

inline void write_bfs_results_meta_json(const std::string& path,
                                        const RunMetadata& meta,
                                        double median_time_ms,
                                        double min_time_ms,
                                        double max_time_ms,
                                        double harmonic_mean_teps,
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
      << "  \"graph\": " << q(meta.graph_path) << ",\n"
      << "  \"num_vertices\": " << meta.num_vertices << ",\n"
      << "  \"num_edges\": " << meta.num_edges << ",\n"
      << "  \"directed\": " << (meta.directed ? "true" : "false") << ",\n"
      << "  \"threads\": " << meta.threads << ",\n"
      << "  \"ranks\": " << meta.ranks << ",\n"
      << "  \"seed\": " << meta.seed << ",\n"
      << "  \"sources\": " << meta.sources << ",\n"
      << "  \"trials\": " << meta.trials << ",\n"
      << "  \"load_time_ms\": " << meta.load_time_ms << ",\n"
      << "  \"timed_trials\": " << timed_trials << ",\n"
      << "  \"median_kernel_time_ms\": " << median_time_ms << ",\n"
      << "  \"min_kernel_time_ms\": " << min_time_ms << ",\n"
      << "  \"max_kernel_time_ms\": " << max_time_ms << ",\n"
      << "  \"harmonic_mean_teps\": " << harmonic_mean_teps << ",\n"
      << "  \"notes\": \"Kernel time excludes graph load/preprocess. "
         "TEPS = edges_examined / time_sec (Graph500-style).\"\n"
      << "}\n";
}

}  // namespace pargraph
