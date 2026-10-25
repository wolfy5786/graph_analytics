#include "graph.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --in <path> --format <edgelist> --out <path.csr> [options]\n"
      << "Options:\n"
      << "  --undirected     insert reverse edges (BFS/CC/TC style)\n"
      << "  --keep-loops     keep self-loops (default: drop)\n"
      << "  --no-dedupe      keep duplicate edges\n"
      << "  --no-sort        do not sort adjacency lists\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path;
  std::string out_path;
  std::string format = "edgelist";
  pargraph::BuildOptions opts;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--in") {
      in_path = need("--in");
    } else if (arg == "--out") {
      out_path = need("--out");
    } else if (arg == "--format") {
      format = need("--format");
    } else if (arg == "--undirected") {
      opts.make_undirected = true;
    } else if (arg == "--keep-loops") {
      opts.drop_self_loops = false;
    } else if (arg == "--no-dedupe") {
      opts.dedupe = false;
    } else if (arg == "--no-sort") {
      opts.sort_adjacency = false;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (in_path.empty() || out_path.empty()) {
    usage(argv[0]);
    return 2;
  }
  if (format != "edgelist") {
    std::cerr << "only --format edgelist is implemented in Phase 0\n";
    return 2;
  }

  try {
    std::cout << "Loading edgelist: " << in_path << "\n";
    pargraph::Graph g = pargraph::load_edgelist_text(in_path, opts);
    std::cout << "Built CSR: V=" << g.num_vertices << " E=" << g.num_edges
              << " directed=" << (g.directed ? "yes" : "no")
              << " sorted=" << (g.sorted ? "yes" : "no") << "\n";

    const auto vr = pargraph::validate_csr(g);
    if (!vr.ok) {
      std::cerr << "CSR validation failed:\n";
      for (const auto& e : vr.errors) {
        std::cerr << "  - " << e << "\n";
      }
      return 1;
    }

    pargraph::save_csr_binary(g, out_path);
    std::cout << "Wrote " << out_path << "\n";

    // Round-trip sanity for large converts.
    pargraph::Graph g2 = pargraph::load_csr_binary(out_path);
    if (!pargraph::graphs_equal(g, g2)) {
      std::cerr << "round-trip mismatch after writing " << out_path << "\n";
      return 1;
    }
    std::cout << "Round-trip OK\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
