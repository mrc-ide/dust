#include <dust/rng.hpp>

#include "test_rng.h"

extern "C" SEXP test_rng() {
  int seed = 1;
  dust::RNG r(1, seed);
  double ret = r.rnorm(0, 0, 1);
  return ScalarReal(ret);
}
