#include <iostream>
#include <dust/random/random.hpp>

void show_numbers(std::vector<double> numbers, const char * label) {
  std::cout << label << ":";
  for (size_t i = 0; i < numbers.size(); ++i) {
    std::cout << " " << numbers[i];
  }
  std::cout << std::endl;
}

int main() {
  using rng_state_type = dust::random::generator<double>;

  auto state = dust::random::seed<rng_state_type>(42);
  size_t n = 10;

  // A place to store numbers:
  std::vector<double> numbers(n);

  for (size_t i = 0; i < n; ++i) {
    numbers[i] = dust::random::uniform<double>(state, 10, 20);
  }
  show_numbers(numbers, "Uniform(10, 20)");

  for (size_t i = 0; i < n; ++i) {
    numbers[i] = dust::random::normal<double>(state, 5, 2);
  }
  show_numbers(numbers, "Normal(5, 2)");

  for (size_t i = 0; i < n; ++i) {
    numbers[i] = dust::random::exponential<double>(state, 6);
  }
  show_numbers(numbers, "Exponential(6)");

  for (size_t i = 0; i < n; ++i) {
    numbers[i] = dust::random::poisson<double>(state, 3);
  }
  show_numbers(numbers, "Poisson(3)");

  for (size_t i = 0; i < n; ++i) {
    numbers[i] = dust::random::binomial<double>(state, 10, 0.3);
  }
  show_numbers(numbers, "Binomial(10, 0.3)");
}
