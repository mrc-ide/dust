#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <algorithm>
#include "xoshiro.hpp"
#include "distr/binomial.hpp"
#include "distr/normal.hpp"
#include "distr/poisson.hpp"
#include "distr/uniform.hpp"

namespace dust {

// This is just a container class for state
template <typename T>
class pRNG {
public:
  pRNG(const size_t n, const std::vector<uint64_t>& seed) :
    n_(n), state_(n * rng_state_t<T>::size()) {
    auto len = rng_state_t<T>::size();
    auto n_seed = seed.size() / len;
    // When the state comes in from R it is *not* interleaved (see
    // also below with export). This is for compatibility with
    // previously used dust runs primarily, and could be relaxed in
    // future.
    for (size_t i = 0; i < n; ++i) {
      rng_state_t<T> s = state(i);
      if (i < n_seed) {
        for (size_t j = 0; j < len; ++j) {
          s[j] = seed[i * len + j];
        }
      } else {
        rng_state_t<T> prev = state(i - 1);
        for (size_t j = 0; j < len; ++j) {
          s[j] = prev[j];
        }
        xoshiro_jump(s);
      }
    }
  }

  size_t size() const {
    return n_;
  }

  void jump() {
    for (size_t i = 0; i < n_; ++i) {
      rng_state_t<T> s = state(i);
      xoshiro_jump(s);
    }
  }

  void long_jump() {
    for (size_t i = 0; i < n_; ++i) {
      rng_state_t<T> s = state(i);
      xoshiro_long_jump(s);
    }
  }

  // Access an individual rng_state by constructing the small struct
  // that contains a pointer to the memory, offset as needed, and our
  // stride.
  rng_state_t<T> state(size_t i) {
    return rng_state_t<T>(state_.data() + i, n_);
  }

  // De-interleave the state on export (to R)
  std::vector<uint64_t> export_state() {
    const auto len = rng_state_t<T>::size();
    std::vector<uint64_t> ret(n_ * len);
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < len; ++j) {
        ret[i * len + j] = state_[i + n_ * j];
      }
    }
    return ret;
  }

  // Imports *non-interleaved* state
  void import_state(const std::vector<uint64_t>& state) {
    const auto len = rng_state_t<T>::size();
    std::vector<uint64_t> ret(n_ * len);
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < len; ++j) {
        state_[i + n_ * j] = state[i * len + j];
      }
    }
  }

private:
  const size_t n_;
  std::vector<uint64_t> state_;
};

}

#endif
