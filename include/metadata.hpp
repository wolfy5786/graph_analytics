#pragma once

#include "results.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace pargraph {

inline std::string utc_timestamp_now() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const std::time_t t = clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

inline std::string detect_hostname() {
  char buf[256] = {};
#if defined(_WIN32)
  DWORD n = static_cast<DWORD>(sizeof(buf));
  if (GetComputerNameA(buf, &n)) {
    return buf;
  }
#else
  if (gethostname(buf, sizeof(buf) - 1) == 0) {
    return buf;
  }
#endif
  return "unknown";
}

inline std::string detect_git_commit() {
  // Best-effort; empty if git unavailable or not a repo.
  FILE* pipe = nullptr;
#if defined(_WIN32)
  pipe = _popen("git rev-parse --short HEAD 2>NUL", "r");
#else
  pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
#endif
  if (!pipe) {
    return "unknown";
  }
  std::array<char, 64> buf{};
  std::string out;
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    out += buf.data();
  }
#if defined(_WIN32)
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  while (!out.empty() &&
         (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
    out.pop_back();
  }
  return out.empty() ? std::string("unknown") : out;
}

inline std::string detect_os() {
#if defined(_WIN32)
  return "Windows";
#elif defined(__APPLE__)
  return "macOS";
#else
  return "Linux";
#endif
}

inline std::string detect_cpu() {
#if defined(_WIN32)
  // Prefer environment override; else a short placeholder filled by runner.
  if (const char* env = std::getenv("PARGRAPH_CPU")) {
    return env;
  }
  return "13th Gen Intel Core i7 (see README machine)";
#else
  if (const char* env = std::getenv("PARGRAPH_CPU")) {
    return env;
  }
  return "unknown";
#endif
}

inline RunMetadata make_run_metadata() {
  RunMetadata m;
  m.timestamp_utc = utc_timestamp_now();
  m.git_commit = detect_git_commit();
  m.hostname = detect_hostname();
  m.os = detect_os();
  m.cpu = detect_cpu();
#if defined(_MSC_VER)
  m.compiler = "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
  m.compiler = "Clang";
#elif defined(__GNUC__)
  m.compiler = "GCC";
#else
  m.compiler = "unknown";
#endif
#if defined(NDEBUG)
  m.cmake_build_type = "Release";
#else
  m.cmake_build_type = "Debug";
#endif
  m.threads = 1;
  m.ranks = 1;
  return m;
}

}  // namespace pargraph
