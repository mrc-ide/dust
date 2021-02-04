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
    export_state(state);
    return state;
  }

  void export_state(std::vector<uint64_t>& state) {
    const size_t n = rng_state_t<T>::size();
    state.resize(size() * n);
    for (size_t i = 0, k = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j, ++k) {
        state[k] = _state[i][j];
      }
    }
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

template <typename T>
device_rng_state_t<T> get_rng_state(const dust::interleaved<uint64_t>& full_rng_state) {
  device_rng_state_t<T> rng_state;
  for (int i = 0; i < device_rng_state_t<T>::size(); i++) {
    rng_state.state[i] = full_rng_state[i];
  }
  return rng_state;
}

// Write state into global memory
template <typename T>
void put_rng_state(device_rng_state_t<T>& rng_state,
                   dust::interleaved<uint64_t>& full_rng_state) {
  for (int i = 0; i < device_rng_state_t<T>::size(); i++) {
    full_rng_state[i] = rng_state.state[i];
  }
}

}

#endif

template <typename T, typename U = typename T::real_t>
U unif_rand(T& state) {
  const uint64_t value = xoshiro_next(state);
  return U(value) / U(std::numeric_limits<uint64_t>::max());
}
