#include <iostream>
#include <dust/random/random.hpp>

int main() {
  using rng_state_type = dust::random::generator<double>;
  auto state = dust::random::seed<rng_state_type>(42);
  for (int i = 0; i < 5; ++i) {
    std::cout << dust::random::random_real<double>(state) << std::endl;
  }
}
