// A simple example showing how to use dust's random number generator
// from other packages.  This example is deliberately very simple and
// does not cover things like persisting state or parallel random
// number generation, however the random.hpp header does include
// support for everything you need but the details will depend
// strongly on your application.
//
// The other requirements (other than this file) are to:
//
// * include `LinkingTo: cpp11, dust` within your DESCRIPTION so that
//   R will find the dust headers when compiling (you do not need to
//   Import anything from either package)
// * make sure your NAMESPACE includes a line like
//   `useDynLib(<package>, .registration = TRUE)`
//   so that the shared library is loaded when your package is.
// * make sure to run cpp11::cpp_register() before compiling your package
//   so that the relevant interfaces are created (R/cpp11.R and
//   cpp11/cpp11.cpp)
#include <cpp11.hpp>
#include <dust/random/random.hpp>

[[cpp11::register]]
cpp11::doubles random_normal(int n, double mu, double sd, int seed) {
  using rng_state_type = dust::random::generator<double>;
  auto state = dust::random::seed<rng_state_type>(seed);

  cpp11::writable::doubles ret(n);
  for (int i = 0; i < n; ++i) {
    ret[i] = dust::random::normal<double>(state, mu, sd);
  }

  return ret;
}
