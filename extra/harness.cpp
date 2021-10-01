#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

#if defined(XOSHIRO256)
typedef uint64_t data_type;
constexpr size_t data_size = 4;
#elif defined(XOSHIRO128)
typedef uint64_t data_type;
constexpr size_t data_size = 4;
#elif defined(XOROSHIRO128)
typedef uint64_t data_type;
constexpr size_t data_size = 4;
#else
#error "no target defined"
#endif

uint64_t splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename T, size_t N>
std::array<T, N> rng_seed(uint64_t seed) {
  std::array<T, N> state;
  const size_t n = N;
  for (size_t i = 0; i < n; ++i) {
    seed = splitmix64(seed);
    state[i] = static_cast<data_type>(seed);
  }
  return state;
}

int main() {
  std::array<data_type, data_size> seed = rng_seed<data_type, data_size>(42);
  for (size_t i = 0; i < data_size; ++i) {
    s[i] = seed[i];
  }
  constexpr int n = 10;
  for (int i = 0; i < n * 3; ++i) {
    if (i == n - 1) {
      jump();
    } else if (i == 2 * n - 1) {
      long_jump();
    }
    auto x = next();
    std::cout <<
      //std::setw(16) << std::setfill('0') << std::hex << x << " " <<
      std::dec << x <<
      std::endl;
  }

  return 0;
}
