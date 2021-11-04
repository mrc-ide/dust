#ifndef DUST_RANDOM_XOSHIRO_STATE_HPP
#define DUST_RANDOM_XOSHIRO_STATE_HPP

#include <array>

#include "dust/random/cuda.hpp"

namespace dust {
namespace random {

// TODO(#306): make this a class enum perhaps
enum xoshiro_mode {STARSTAR, PLUSPLUS, PLUS};

// Common storage for all generators
template <typename T, size_t N, xoshiro_mode M>
struct xoshiro_state {
  typedef T int_type;
  HOSTDEVICE static constexpr size_t size() {
    return N;
  }
  int_type state[N];
  // This flag indicates that the distributions should return the
  // deterministic expectation of the draw, and not use any random
  // numbers
  bool deterministic = false;
  HOSTDEVICE int_type& operator[](size_t i) {
    return state[i];
  }
};

template <typename T>
typename T::int_type next(T& state);

template <typename T, size_t N, xoshiro_mode M>
std::array<T, N> jump_constants();

template <typename T, size_t N, xoshiro_mode M>
std::array<T, N> long_jump_constants();

}
}

#endif
