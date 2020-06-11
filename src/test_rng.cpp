#include <dust/rng.hpp>
#include <dust/util.hpp>

#include "test_rng.h"

extern "C" SEXP test_rng(SEXP r_n, SEXP r_seed, SEXP r_n_thread) {
  const int n = dust::util::as_size(r_n, "n");
  const int seed = dust::util::as_size(r_seed, "seed");
  const int n_thread = dust::util::as_size(r_n_thread, "n_thread");

  const double mean = 0, sd = 1;

  dust::RNG r(n_thread, seed);

  SEXP ret = PROTECT(allocVector(REALSXP, n));
  double * y = REAL(ret);
  for (int i = 0; i < n; ++i) {
    y[i] = r.rnorm(i % n_thread, mean, sd);
  }

  UNPROTECT(1);
  return ret;
}
