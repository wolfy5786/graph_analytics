#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pargraph {

class Timer {
 public:
  void start() { start_ = Clock::now(); }
  void stop() { end_ = Clock::now(); running_ = false; }

  double elapsed_sec() const {
    const auto end = running_ ? Clock::now() : end_;
    return std::chrono::duration<double>(end - start_).count();
  }

  double elapsed_ms() const { return elapsed_sec() * 1e3; }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_{};
  Clock::time_point end_{};
  bool running_ = true;
};

// Simple stats helpers for benchmark reporting.
inline double median_sorted(std::vector<double> xs) {
  if (xs.empty()) {
    return 0.0;
  }
  const std::size_t n = xs.size();
  // caller may pass unsorted — sort copy
  // already named median_sorted but we'll sort to be safe
  for (std::size_t i = 1; i < n; ++i) {
    double key = xs[i];
    std::size_t j = i;
    while (j > 0 && xs[j - 1] > key) {
      xs[j] = xs[j - 1];
      --j;
    }
    xs[j] = key;
  }
  if (n % 2 == 1) {
    return xs[n / 2];
  }
  return 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

inline double harmonic_mean(const std::vector<double>& xs) {
  if (xs.empty()) {
    return 0.0;
  }
  double inv_sum = 0.0;
  std::size_t n = 0;
  for (double x : xs) {
    if (x > 0.0 && std::isfinite(x)) {
      inv_sum += 1.0 / x;
      ++n;
    }
  }
  if (n == 0 || inv_sum <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(n) / inv_sum;
}

}  // namespace pargraph
