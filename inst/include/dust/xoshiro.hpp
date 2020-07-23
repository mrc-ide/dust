#ifndef DUST_XOSHIRO_HPP
#define DUST_XOSHIRO_HPP

#include <array>
#include <cstdint>
#include <vector>
#include <limits>

// This is derived from http://prng.di.unimi.it/xoshiro256starstar.c
// and http://prng.di.unimi.it/splitmix64.c, copies of which are
// included in the package (in inst/rng in the source package). The
// original code is CC0 licenced but was written by David Blackman and
// Sebastiano Vigna.
//
// MD5 (splitmix64.c) = 7e38529aa7bb36624ae4a9d6808e7e3f
// MD5 (xoshiro256starstar.c) = 05f9ecd49bbed98304d982313c91d0f6

#define XOSHIRO_WIDTH 4

namespace dust {

constexpr size_t xoshiro_size() {
  return 4;
}

// It is not really clear what the best way of storing the state is;
// we could store it interleaved or adjacent. For now, let's use
// adjacent. To do interleaved we just need to know the stride here,
// so could store a struct with *uint64_t state and size_t stride
template <typename T>
struct rng_state_t {
  typedef T real_t;
  static size_t size() {
    return 4;
  }
  std::array<uint64_t, 4> state;
  uint64_t& operator[](size_t i) {
    return state[i];
  }
};

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// This is the core generator (next() in the original C code)
template <typename T>
inline uint64_t xoshiro_next(rng_state_t<T>& state) {
  const uint64_t result = rotl(state[1] * 5, 7) * 9;

  const uint64_t t = state[1] << 17;

  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];

  state[2] ^= t;

  state[3] = rotl(state[3], 45);

  return result;
}

inline uint64_t splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename T>
inline void xoshiro_set_seed(rng_state_t<T>& state, uint64_t seed) {
  // normal brain: for i in 1:4
  // advanced brain: -funroll-loops
  // galaxy brain:
  state[0] = splitmix64(seed);
  state[1] = splitmix64(state[0]);
  state[2] = splitmix64(state[1]);
  state[3] = splitmix64(state[2]);
}

/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
template <typename T>
inline void xoshiro_jump(rng_state_t<T>& state) {
  static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                   0xa9582618e03fc9aa, 0x39abdc4529b1661c };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (long unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
    for (int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= state[0];
        s1 ^= state[1];
        s2 ^= state[2];
        s3 ^= state[3];
      }
      xoshiro_next(state);
    }
  }

  state[0] = s0;
  state[1] = s1;
  state[2] = s2;
  state[3] = s3;
}

/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */
template <typename T>
inline void xoshiro_long_jump(rng_state_t<T>& state) {
  static const uint64_t LONG_JUMP[] =
    { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
      0x77710069854ee241, 0x39109bb02acbe635 };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (long unsigned int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++) {
    for (int b = 0; b < 64; b++) {
      if (LONG_JUMP[i] & UINT64_C(1) << b) {
        s0 ^= state[0];
        s1 ^= state[1];
        s2 ^= state[2];
        s3 ^= state[3];
      }
      xoshiro_next(state);
    }
  }

  state[0] = s0;
  state[1] = s1;
  state[2] = s2;
  state[3] = s3;
}

/*
class Xoshiro {
public:
  // Definitions to satisfy interface of URNG in C++11
  typedef uint64_t result_type;
  static constexpr uint64_t min() {
    return std::numeric_limits<uint64_t>::min();
  }
  static constexpr uint64_t max() {
    return std::numeric_limits<uint64_t>::max();
  }
  uint64_t operator()() {
    return xoshiro_next(_state);
  }

  Xoshiro(uint64_t seed) {
    set_seed(seed);
  }

  // Change internal state
  void set_seed(uint64_t seed) {
    xoshiro_set_seed(_state, seed);
  }

  void jump() {
    xoshiro_jump(_state);
  }

  void long_jump() {
    xoshiro_long_jump(_state);
  }

  uint64_t* state() {
    return _state.data();
  }

  void get_state(std::vector<uint64_t>& state) {
    for (size_t i = 0; i < XOSHIRO_WIDTH; ++i) {
      state.push_back(_state[i]);
    }
  }

private:
  std::array<uint64_t, 4> _state;
};
*/

template <typename real_t>
real_t unif_rand(rng_state_t<real_t>& state) {
  const uint64_t value = xoshiro_next(state);
  return real_t(value) / real_t(std::numeric_limits<uint64_t>::max());
}

}

#endif
