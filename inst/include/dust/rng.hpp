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
class pRNG { // # nocov
public:
  pRNG(const size_t n, const std::vector<uint64_t>& seed) {
    rng_state_t<T> s;
    auto len = rng_state_t<T>::size();
    auto n_seed = seed.size() / len;
    for (size_t i = 0; i < n; ++i) {
      if (i < n_seed) {
        std::copy_n(seed.begin() + i * len, len, s.state.begin());
      } else {
        xoshiro_jump(s);
      }
      _state.push_back(s);
    }
  }

  size_t size() const {
    return _state.size();
  }

  void jump() {
    for (size_t i = 0; i < _state.size(); ++i) {
      xoshiro_jump(_state[i]);
    }
  }

  void long_jump() {
    for (size_t i = 0; i < _state.size(); ++i) {
      xoshiro_long_jump(_state[i]);
    }
  }

  rng_state_t<T>& state(size_t i) {
    return _state[i];
  }

  std::vector<uint64_t> export_state() {
    std::vector<uint64_t> state;
    const size_t n = rng_state_t<T>::size();
    state.reserve(size() * n);
    for (size_t i = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j) {
        state.push_back(_state[i][j]);
      }
    }
    return state;
  }

  void import_state(const std::vector<uint64_t>& state) {
    auto it = state.begin();
    const size_t n = rng_state_t<T>::size();
    for (size_t i = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j) {
        _state[i][j] = *it;
        ++it;
      }
    }
  }

private:
  std::vector<rng_state_t<T>> _state;
};

}

#endif
