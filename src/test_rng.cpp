#include <dust/rng.hpp>

#include "test_rng.h"

extern "C" SEXP test_rng(SEXP r_n, SEXP r_seed) {
  int n = INTEGER(r_n)[0];
  int seed = INTEGER(r_seed)[0];

  const int nt = 1, index = 0;
  const double mean = 0, sd = 1;

  dust::RNG r(nt, seed);

  SEXP ret = PROTECT(allocVector(REALSXP, n));
  double * y = REAL(ret);
  for (int i = 0; i < n; ++i) {
    y[i] = r.rnorm(index, mean, sd);
  }

  UNPROTECT(1);
  return ret;
}
