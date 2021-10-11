#include <iostream>
#include <vector>
// #include <omp.h>
#include <dust/random/random.hpp>

template <typename real_type>
std::vector<real_type> random_sum(int n_generators, int n_draws,
                                  int seed, int n_threads) {
  using rng_state_type = dust::random::generator<real_type>;
  dust::random::prng<rng_state_type> rng(n_generators, seed, false);

  std::vector<real_type> ret(n_generators, 0.0);

  #pragma omp parallel for schedule(static) num_threads(n_threads)
  for (int i = 0; i < n_generators; ++i) {
    for (size_t j = 0; j < n_draws; ++j) {
      ret[i] += dust::random::random_real<real_type>(rng.state(i));
    }
  }

  return ret;
}

int main(int argc, char* argv[]) {
  // Extremely simple but non-robust arg handling:
  if (argc < 2) {
    std::cout <<
      "Usage: rnguse <n_draws> [<n_generators> [<seed> [<n_threads>]]]" <<
      std::endl;
    return 1;
  }
  int n_draws      = atoi(argv[1]);
  int n_generators = argc < 3 ?  10 : atoi(argv[2]);
  int seed         = argc < 4 ?  42 : atoi(argv[3]);
  int n_threads    = argc < 5 ?   1 : atoi(argv[4]);

  auto ans = random_sum<double>(n_generators, n_draws, seed, n_threads);

  for (int i = 0; i < n_generators; ++i) {
    std::cout << i << ": " << ans[i] << std::endl;
  }

  return 0;
}
