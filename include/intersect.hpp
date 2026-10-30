#pragma once

#include "graph.hpp"

#include <cstddef>
#include <cstdint>

namespace pargraph {

// Sorted ascending neighbor-range intersection size (merge walk).
// Both ranges must be sorted ascending. Returns |A ∩ B|.
inline std::uint64_t intersect_merge(const vtx_t* a, const vtx_t* a_end,
                                     const vtx_t* b, const vtx_t* b_end) {
  std::uint64_t count = 0;
  while (a != a_end && b != b_end) {
    if (*a == *b) {
      ++count;
      ++a;
      ++b;
    } else if (*a < *b) {
      ++a;
    } else {
      ++b;
    }
  }
  return count;
}

// Galloping (exponential search) intersection for skewed degree pairs.
// Advances the denser cursor with doubling probes, then binary-searches.
inline std::uint64_t intersect_galloping(const vtx_t* a, const vtx_t* a_end,
                                         const vtx_t* b, const vtx_t* b_end) {
  std::uint64_t count = 0;
  while (a != a_end && b != b_end) {
    if (*a == *b) {
      ++count;
      ++a;
      ++b;
      continue;
    }
    if (*a < *b) {
      // Gallop a forward toward *b.
      const vtx_t* lo = a;
      std::size_t step = 1;
      while (lo + step < a_end && *(lo + step) < *b) {
        lo += step;
        step <<= 1;
      }
      const vtx_t* hi = (lo + step < a_end) ? lo + step : a_end;
      // Binary search [*lo, *hi) for first >= *b.
      while (lo < hi) {
        const vtx_t* mid = lo + (hi - lo) / 2;
        if (*mid < *b) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      a = lo;
    } else {
      const vtx_t* lo = b;
      std::size_t step = 1;
      while (lo + step < b_end && *(lo + step) < *a) {
        lo += step;
        step <<= 1;
      }
      const vtx_t* hi = (lo + step < b_end) ? lo + step : b_end;
      while (lo < hi) {
        const vtx_t* mid = lo + (hi - lo) / 2;
        if (*mid < *a) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      b = lo;
    }
  }
  return count;
}

enum class IntersectMode : std::uint8_t { Merge = 0, Galloping = 1 };

inline std::uint64_t intersect_sorted(const vtx_t* a, const vtx_t* a_end,
                                      const vtx_t* b, const vtx_t* b_end,
                                      IntersectMode mode) {
  if (mode == IntersectMode::Galloping) {
    // Prefer galloping from the shorter list into the longer one.
    if (static_cast<std::size_t>(a_end - a) >
        static_cast<std::size_t>(b_end - b)) {
      return intersect_galloping(b, b_end, a, a_end);
    }
    return intersect_galloping(a, a_end, b, b_end);
  }
  return intersect_merge(a, a_end, b, b_end);
}

}  // namespace pargraph
