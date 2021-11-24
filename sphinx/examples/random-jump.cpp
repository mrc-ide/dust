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
  using rng_state_type = dust::random::xoroshiro128plusplus;

  auto state = dust::random::seed<rng_state_type>(42);
  show_state(state, "Initial state");

  dust::random::jump(state);
  show_state(state, "\nAfter one jump");

  dust::random::long_jump(state);
  show_state(state, "\nAfter long jump");
}
