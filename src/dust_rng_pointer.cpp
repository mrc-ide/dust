#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cpp11/external_pointer.hpp>
#include <dust/r/helpers.hpp>
#include <dust/r/random.hpp>

[[cpp11::register]]
cpp11::sexp dust_rng_pointer_init(int n_streams, cpp11::sexp seed,
                                  int long_jump, std::string algorithm) {
  cpp11::sexp ret;

  dust::r::validate_size(long_jump, "long_jump");

  using namespace dust::random;
  if (algorithm == "xoshiro256starstar") {
    ret = r::rng_pointer_init<xoshiro256starstar>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro256plusplus") {
    ret = r::rng_pointer_init<xoshiro256plusplus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro256plus") {
    ret = r::rng_pointer_init<xoshiro256plus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro128starstar") {
    ret = r::rng_pointer_init<xoshiro128starstar>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro128plusplus") {
    ret = r::rng_pointer_init<xoshiro128plusplus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro128plus") {
    ret = r::rng_pointer_init<xoshiro128plus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoroshiro128starstar") {
    ret = r::rng_pointer_init<xoroshiro128starstar>(n_streams, seed, long_jump);
  } else if (algorithm == "xoroshiro128plusplus") {
    ret = r::rng_pointer_init<xoroshiro128plusplus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoroshiro128plus") {
    ret = r::rng_pointer_init<xoroshiro128plus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro512starstar") {
    ret = r::rng_pointer_init<xoshiro512starstar>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro512plusplus") {
    ret = r::rng_pointer_init<xoshiro512plusplus>(n_streams, seed, long_jump);
  } else if (algorithm == "xoshiro512plus") {
    ret = r::rng_pointer_init<xoshiro512plus>(n_streams, seed, long_jump);
  } else {
    cpp11::stop("Unknown algorithm '%s'", algorithm.c_str());
  }

  return ret;
}

[[cpp11::register]]
void dust_rng_pointer_sync(cpp11::environment obj, std::string algorithm) {
  using namespace dust::random;
  if (algorithm == "xoshiro256starstar") {
    r::rng_pointer_sync<xoshiro256starstar>(obj);
  } else if (algorithm == "xoshiro256plusplus") {
    r::rng_pointer_sync<xoshiro256plusplus>(obj);
  } else if (algorithm == "xoshiro256plus") {
    r::rng_pointer_sync<xoshiro256plus>(obj);
  } else if (algorithm == "xoshiro128starstar") {
    r::rng_pointer_sync<xoshiro128starstar>(obj);
  } else if (algorithm == "xoshiro128plusplus") {
    r::rng_pointer_sync<xoshiro128plusplus>(obj);
  } else if (algorithm == "xoshiro128plus") {
    r::rng_pointer_sync<xoshiro128plus>(obj);
  } else if (algorithm == "xoroshiro128starstar") {
    r::rng_pointer_sync<xoroshiro128starstar>(obj);
  } else if (algorithm == "xoroshiro128plusplus") {
    r::rng_pointer_sync<xoroshiro128plusplus>(obj);
  } else if (algorithm == "xoroshiro128plus") {
    r::rng_pointer_sync<xoroshiro128plus>(obj);
  } else if (algorithm == "xoshiro512starstar") {
    r::rng_pointer_sync<xoshiro512starstar>(obj);
  } else if (algorithm == "xoshiro512plusplus") {
    r::rng_pointer_sync<xoshiro512plusplus>(obj);
  } else if (algorithm == "xoshiro512plus") {
    r::rng_pointer_sync<xoshiro512plus>(obj);
  }
}

// This exists to check some error paths in rng_pointer_get; it is not
// for use by users.
[[cpp11::register]]
double test_rng_pointer_get(cpp11::environment obj, int n_streams) {
  using namespace dust::random;
  auto rng = r::rng_pointer_get<xoshiro256plus>(obj, n_streams);
  return random_real<double>(rng->state(0));
}
