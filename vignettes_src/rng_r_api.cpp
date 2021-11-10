#include <cpp11.hpp>
#include <R_ext/Random.h>

[[cpp11::register]]
double pi_r_api(int n) {
  int tot = 0;
  GetRNGstate();
  for (int i = 0; i < n; ++i) {
    const double u1 = unif_rand();
    const double u2 = unif_rand();
    if (u1 * u1 + u2 * u2 < 1) {
      tot++;
    }
  }
  PutRNGstate();
  return tot / static_cast<double>(n) * 4.0;
}
