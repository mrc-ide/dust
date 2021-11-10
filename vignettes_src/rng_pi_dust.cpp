#include <cpp11.hpp>
#include <dust/r/random.hpp>

[[cpp11::linking_to(dust)]]
[[cpp11::register]]
double pi_dust(int n, cpp11::sexp ptr) {
  auto rng =
    dust::random::r::rng_pointer_get<dust::random::xoshiro256plus_state>(ptr);
  auto& state = rng->state(0);
  int tot = 0;
  for (int i = 0; i < n; ++i) {
    const double u1 = dust::random::random_real<double>(state);
    const double u2 = dust::random::random_real<double>(state);
    if (u1 * u1 + u2 * u2 < 1) {
      tot++;
    }
  }
  return tot / static_cast<double>(n) * 4.0;
}
