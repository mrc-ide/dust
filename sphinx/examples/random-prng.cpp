#include <iostream>
#include <dust/random/random.hpp>

template <typename T>
void show_state(T& state, const char * label) {
  std::cout << label << std::endl;
  std::cout << std::hex;
  for (size_t i = 0; i < T::size(); ++i) {
    std::cout << state[i] << std::endl;
  }
}

int main() {
  // Using a small state so it's easier to see the output
  using rng_state_type = dust::random::xoroshiro128plus;

  // Create a prng object with 4 streams (and initial seed 42)
  auto obj = dust::random::prng<rng_state_type>(4, 42);
  show_state(obj.state(0), "first stream");

  // An equivalently created state
  auto cmp = dust::random::seed<rng_state_type>(42);
  show_state(cmp, "\nequivalent state");

  // The second stream
  show_state(obj.state(1), "\nsecond stream");

  // Jumping our state forward gets to the same place
  dust::random::jump(cmp);
  show_state(cmp, "\nequivalent state");
}
