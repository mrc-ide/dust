#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cpp11/external_pointer.hpp>
#include <dust/r/random.hpp>

[[cpp11::register]]
cpp11::sexp dust_rng_pointer_init(int n_streams, cpp11::sexp seed,
                                  std::string algorithm) {
  cpp11::sexp ret;

  using namespace dust::random;
  if (algorithm == "xoshiro256starstar") {
    ret = r::rng_pointer_init<xoshiro256starstar_state>(n_streams, seed);
  } else if (algorithm == "xoshiro256plusplus") {
    ret = r::rng_pointer_init<xoshiro256plusplus_state>(n_streams, seed);
  } else if (algorithm == "xoshiro256plus") {
    ret = r::rng_pointer_init<xoshiro256plus_state>(n_streams, seed);
  } else if (algorithm == "xoshiro128starstar") {
    ret = r::rng_pointer_init<xoshiro128starstar_state>(n_streams, seed);
  } else if (algorithm == "xoshiro128plusplus") {
    ret = r::rng_pointer_init<xoshiro128plusplus_state>(n_streams, seed);
  } else if (algorithm == "xoshiro128plus") {
    ret = r::rng_pointer_init<xoshiro128plus_state>(n_streams, seed);
  } else if (algorithm == "xoroshiro128starstar") {
    ret = r::rng_pointer_init<xoroshiro128starstar_state>(n_streams, seed);
  } else if (algorithm == "xoroshiro128plusplus") {
    ret = r::rng_pointer_init<xoroshiro128plusplus_state>(n_streams, seed);
  } else if (algorithm == "xoroshiro128plus") {
    ret = r::rng_pointer_init<xoroshiro128plus_state>(n_streams, seed);
  } else if (algorithm == "xoshiro512starstar") {
    ret = r::rng_pointer_init<xoshiro512starstar_state>(n_streams, seed);
  } else if (algorithm == "xoshiro512plusplus") {
    ret = r::rng_pointer_init<xoshiro512plusplus_state>(n_streams, seed);
  } else if (algorithm == "xoshiro512plus") {
    ret = r::rng_pointer_init<xoshiro512plus_state>(n_streams, seed);
  } else {
    cpp11::stop("Unknown algorithm '%s'", algorithm.c_str());
  }

  return ret;
}

[[cpp11::register]]
void dust_rng_pointer_sync(cpp11::environment obj, std::string algorithm) {
  using namespace dust::random;
  if (algorithm == "xoshiro256starstar") {
    r::rng_pointer_sync<xoshiro256starstar_state>(obj);
  } else if (algorithm == "xoshiro256plusplus") {
    r::rng_pointer_sync<xoshiro256plusplus_state>(obj);
  } else if (algorithm == "xoshiro256plus") {
    r::rng_pointer_sync<xoshiro256plus_state>(obj);
  } else if (algorithm == "xoshiro128starstar") {
    r::rng_pointer_sync<xoshiro128starstar_state>(obj);
  } else if (algorithm == "xoshiro128plusplus") {
    r::rng_pointer_sync<xoshiro128plusplus_state>(obj);
  } else if (algorithm == "xoshiro128plus") {
    r::rng_pointer_sync<xoshiro128plus_state>(obj);
  } else if (algorithm == "xoroshiro128starstar") {
    r::rng_pointer_sync<xoroshiro128starstar_state>(obj);
  } else if (algorithm == "xoroshiro128plusplus") {
    r::rng_pointer_sync<xoroshiro128plusplus_state>(obj);
  } else if (algorithm == "xoroshiro128plus") {
    r::rng_pointer_sync<xoroshiro128plus_state>(obj);
  } else if (algorithm == "xoshiro512starstar") {
    r::rng_pointer_sync<xoshiro512starstar_state>(obj);
  } else if (algorithm == "xoshiro512plusplus") {
    r::rng_pointer_sync<xoshiro512plusplus_state>(obj);
  } else if (algorithm == "xoshiro512plus") {
    r::rng_pointer_sync<xoshiro512plus_state>(obj);
  } else {
    cpp11::stop("Unknown algorithm '%s'", algorithm.c_str());
  }
}
