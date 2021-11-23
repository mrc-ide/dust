#ifndef DUST_RANDOM_XOSHIRO_STATE_HPP
#define DUST_RANDOM_XOSHIRO_STATE_HPP

#include <array>

#include "dust/random/cuda_compatibility.hpp"

namespace dust {
namespace random {

enum class scrambler {
                      starstar, ///< Two multiplications
                      plusplus, ///< Two additions
                      plus      ///< One addition
};

/// Basic struct to hold random number state for a single stream
/// @tparam T The integer type (``uint32_t`` or ``uint64_t``)
/// @tparam N The number of integers included in the state (2, 4, or 8)
/// @tparam X The scrambler type
template <typename T, size_t N, scrambler X>
class xoshiro_state {
public:
  /// Type alias used to find the integer type
  using int_type = T;
  /// Static method, returning the number of integers per state
  __host__ __device__ static constexpr size_t size() {
    return N;
  }
  /// Array of state
  int_type state[N];
  /// This flag indicates that the distributions should return the
  /// deterministic expectation of the draw, and not use any random
  /// numbers
  bool deterministic = false;
  /// Accessor method, used to both get and set the underlying state
  __host__ __device__ int_type& operator[](size_t i) {
    return state[i];
  }
};

template <typename T>
typename T::int_type next(T& state);

template <typename T>
std::array<typename T::int_type, T::size()> jump_constants();

template <typename T>
std::array<typename T::int_type, T::size()> long_jump_constants();

}
}

#endif
