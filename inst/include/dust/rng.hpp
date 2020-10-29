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
    // TODO: This logic probably comes out into something easier to
    // look at.
    auto len = rng_state_t<T>::size();
    auto n_seed = seed.size() / len;
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
      xoshiro_jump(state(i));
    }
  }

  void long_jump() {
    for (size_t i = 0; i < n_; ++i) {
      xoshiro_long_jump(state(i));
    }
  }

  // Access an individual rng_state by constructing the small struct
  // that contains a pointer to the memory, offset as needed, and our
  // stride.
  rng_state_t<T> state(size_t i) {
    return rng_state_t<T>(state_.data() + i, n_);
  }

  // Possibly nicer way of doing the above
  rng_state_t<T> operator[](size_t i) {
    return rng_state_t<T>(state_.data() + i * n_, n_);
  }

  // We might make this optionally dump out the raw state, at least
  // for debugging?
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
