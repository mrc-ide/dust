#include <dust/rng.hpp>
#include <dust/util.hpp>

#include "test_rng.h"

extern "C" SEXP test_rng(SEXP r_n, SEXP r_seed, SEXP r_n_generators) {
  const int n = dust::util::as_size(r_n, "n");
  const int seed = dust::util::as_size(r_seed, "seed");
  const int n_generators = dust::util::as_size(r_n_generators, "n_generators");

  const double mean = 0, sd = 1;

  dust::pRNG r(n_generators, seed);

  SEXP ret = PROTECT(allocVector(REALSXP, n));
  double * y = REAL(ret);
  for (int i = 0; i < n; ++i) {
    y[i] = r(i % n_generators).rnorm(mean, sd);
  }

  UNPROTECT(1);
  return ret;
}


extern "C" SEXP test_rng_rbinom(SEXP r_n, SEXP r_p,
                                SEXP r_seed, SEXP r_n_generators) {
  const int seed = dust::util::as_size(r_seed, "seed");
  const int n_generators = dust::util::as_size(r_n_generators, "n_generators");

  size_t n_samples = length(r_n);
  int *n = INTEGER(r_n);
  double *p = REAL(r_p);

  dust::pRNG r(n_generators, seed);

  SEXP ret = PROTECT(allocVector(INTSXP, n_samples));
  int * y = INTEGER(ret);
  for (size_t i = 0; i < n_samples; ++i) {
    y[i] = r(i % n_generators).rbinom(n[i], p[i]);
  }

  UNPROTECT(1);
  return ret;
}
