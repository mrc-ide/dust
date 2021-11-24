#include <iostream>
#include <dust/random/random.hpp>

int main() {
  using rng_state_type = dust::random::generator<double>;
  auto state = dust::random::seed<rng_state_type>(42);

  std::cout << "uniform: " <<
    dust::random::random_real<double>(state) << std::endl;
  std::cout << "normal: " <<
    dust::random::random_normal<double>(state) << std::endl;
  std::cout << "unsigned integer: " <<
    dust::random::random_int<uint32_t>(state) << std::endl;
  std::cout << "signed integer: " <<
    dust::random::random_int<int16_t>(state) << std::endl;
}
