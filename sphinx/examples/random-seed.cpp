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

template <typename T>
void draw_numbers(T& state, int n, const char * label) {
  std::cout << label << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << dust::random::random_normal<double>(state) << std::endl;
  }
}

int main() {
  using rng_state_type = dust::random::generator<double>;
  // construct a state as a value
  auto state = dust::random::seed<rng_state_type>(1);

  show_state(state, "Initial state");
  draw_numbers(state, 5, "\nRandom samples");

  // construct a state in place
  dust::random::seed(state, 2);
  show_state(state, "\nState with slightly different seed");
  draw_numbers(state, 5, "\nRandom samples");
}
