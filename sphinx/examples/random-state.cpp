#include <iostream>
#include <dust/random/random.hpp>

int main() {
  using rng_state_type = dust::random::xoroshiro128plusplus;
  std::cout << "Using " << rng_state_type::size() << " " <<
    std::numeric_limits<rng_state_type::int_type>::digits <<
    " bit unsigned integers" << std::endl;

  auto state = dust::random::seed<rng_state_type>(42);
  std::cout << std::hex << state[0] << " " << state[1] << std::endl;
}
