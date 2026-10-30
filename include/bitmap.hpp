#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pargraph {

// Dense bitset used as a BFS frontier (and similar set membership tests).
// Word-parallel scans make bottom-up BFS cheap when the frontier is large.
class Bitmap {
 public:
  Bitmap() = default;

  explicit Bitmap(std::size_t nbits) { reset(nbits); }

  void reset(std::size_t nbits) {
    nbits_ = nbits;
    words_.assign((nbits + 63) / 64, 0);
  }

  void clear() { std::fill(words_.begin(), words_.end(), 0ull); }

  std::size_t size() const { return nbits_; }

  std::size_t word_count() const { return words_.size(); }

  const std::uint64_t* data() const { return words_.data(); }

  std::uint64_t* data() { return words_.data(); }

  bool get(std::size_t i) const {
    return (words_[i >> 6] >> (i & 63)) & 1ull;
  }

  void set(std::size_t i) { words_[i >> 6] |= (1ull << (i & 63)); }

  void reset_bit(std::size_t i) { words_[i >> 6] &= ~(1ull << (i & 63)); }

  // True if any bit is set. O(words).
  bool any() const {
    for (std::uint64_t w : words_) {
      if (w != 0) {
        return true;
      }
    }
    return false;
  }

  std::uint64_t count() const {
    std::uint64_t c = 0;
    for (std::uint64_t w : words_) {
      // Portable popcount (no reliance on GCC/Clang builtins / MSVC intrinsics).
      std::uint64_t x = w;
      x = x - ((x >> 1) & 0x5555555555555555ull);
      x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
      x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
      c += (x * 0x0101010101010101ull) >> 56;
    }
    return c;
  }

 private:
  std::size_t nbits_ = 0;
  std::vector<std::uint64_t> words_;
};

}  // namespace pargraph
