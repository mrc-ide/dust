#include <cpp11.hpp>
#include <dust/r/random.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

[[cpp11::linking_to(dust)]]
[[cpp11::register]]
double pi_dust_parallel(int n, cpp11::sexp ptr, int n_threads) {
  auto rng =
    dust::random::r::rng_pointer_get<dust::random::xoshiro256plus_state>(ptr);
  const auto n_streams = rng->size();
  int tot = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)  num_threads(n_threads) \
  reduction(+:tot)
#endif
  for (size_t i = 0; i < n_streams; ++i) {
    auto& state = rng->state(0);
    int tot_i = 0;
    for (int i = 0; i < n; ++i) {
      const double u1 = dust::random::random_real<double>(state);
      const double u2 = dust::random::random_real<double>(state);
      if (u1 * u1 + u2 * u2 < 1) {
        tot_i++;
      }
    }
    tot += tot_i;
  }
  return tot / static_cast<double>(n * n_streams) * 4.0;
}
