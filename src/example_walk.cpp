#include <dust/dust.hpp>
#include <R.h>
#include <Rinternals.h>

#include "example_walk.h"

class walk {
public:
  walk(SEXP data) : sd(REAL(data)[0]) {
  }
  size_t size() const {
    return 1;
  }
  void update(size_t step, const std::vector<double> state, dust::RNG rng,
              const size_t thread_idx, std::vector<double>& state_next) {
    double mean = state[0];
    state_next[0] = rng.rnorm(thread_idx, mean, sd);
  }
private:
  double sd;
};


extern "C" SEXP test_walk(SEXP sd, SEXP r_n_particles, SEXP r_seed) {
  size_t n_particles = REAL(r_n_particles)[0];
  size_t seed = REAL(r_seed)[0];

  std::vector<size_t> index_y = {0};
  size_t n_threads = 1;

  dust::Dust<walk> d(sd, index_y, n_threads, seed, n_particles);

  return R_NilValue;
}
