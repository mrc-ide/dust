#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

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
  pRNG(const size_t n, const uint64_t seed) {
    rng_state_t<T> s;
    xoshiro_set_seed(s, seed);

    _state.push_back(s);
    for (size_t i = 1; i < n; ++i) {
      xoshiro_jump(s);
      _state.push_back(s);
    }
  }

  // Initialise from a vector of seed
  pRNG(const size_t n, const std::vector<uint64_t> seed) {
    rng_state_t<T> s;
    auto it = seed.begin();
    auto len = rng_state_t<T>::size();
    for (size_t i = 0; i < n; ++i) {
      if (it == seed.end()) {
        xoshiro_jump(s);
      } else {
        std::copy_n(it, len, s.state.begin());
        it += len;
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

private:
  std::vector<rng_state_t<T>> _state;
};

}

#endif
