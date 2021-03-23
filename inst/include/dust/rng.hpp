#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <algorithm>
#include "xoshiro.hpp"
#include "distr/binomial.hpp"
#include "distr/exponential.hpp"
#include "distr/normal.hpp"
#include "distr/poisson.hpp"
#include "distr/uniform.hpp"
#include "containers.hpp"

namespace dust {

// This is just a container class for state
template <typename T>
class pRNG { // # nocov
public:
  pRNG(const size_t n, const std::vector<uint64_t>& seed) {
    rng_state_t<T> s;
    auto len = rng_state_t<T>::size();
    auto n_seed = seed.size() / len;
    for (size_t i = 0; i < n; ++i) {
      if (i < n_seed) {
        std::copy_n(seed.begin() + i * len, len, std::begin(s.state));
      } else {
        xoshiro_jump(s);
      }
      state_.push_back(s);
    }
  }

  size_t size() const {
    return state_.size();
  }

  void jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      xoshiro_jump(state_[i]);
    }
  }

  void long_jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      xoshiro_long_jump(state_[i]);
    }
  }

  rng_state_t<T>& state(size_t i) {
    return state_[i];
  }

  std::vector<uint64_t> export_state() {
    std::vector<uint64_t> state;
    export_state(state);
    return state;
  }

  void export_state(std::vector<uint64_t>& state) {
    const size_t n = rng_state_t<T>::size();
    state.resize(size() * n);
    for (size_t i = 0, k = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j, ++k) {
        state[k] = state_[i][j];
      }
    }
  }

  void import_state(const std::vector<uint64_t>& state) {
    auto it = state.begin();
    const size_t n = rng_state_t<T>::size();
    for (size_t i = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j) {
        state_[i][j] = *it;
        ++it;
      }
    }
  }

private:
  std::vector<rng_state_t<T>> state_;
};

template <typename T>
DEVICE rng_state_t<T> get_rng_state(const dust::interleaved<uint64_t>& full_rng_state) {
  rng_state_t<T> rng_state;
  for (size_t i = 0; i < rng_state.size(); i++) {
    rng_state.state[i] = full_rng_state[i];
  }
  return rng_state;
}

// Write state into global memory
template <typename T>
DEVICE void put_rng_state(rng_state_t<T>& rng_state,
                   dust::interleaved<uint64_t>& full_rng_state) {
  for (size_t i = 0; i < rng_state.size(); i++) {
    full_rng_state[i] = rng_state.state[i];
  }
}

}

#endif
