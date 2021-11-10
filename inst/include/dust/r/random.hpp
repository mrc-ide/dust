#ifndef DUST_R_RANDOM_HPP
#define DUST_R_RANDOM_HPP

#include <cstring> // memcpy

#include <cpp11/raws.hpp>
#include <R_ext/Random.h>

#include "dust/random/generator.hpp"
#include "dust/random/prng.hpp"

namespace dust {
namespace random {
namespace r {

template <typename rng_state_type>
std::vector<typename rng_state_type::int_type> as_rng_seed(cpp11::sexp r_seed) {
  typedef typename rng_state_type::int_type int_type;
  auto seed_type = TYPEOF(r_seed);
  std::vector<int_type> seed;
  if (seed_type == INTSXP || seed_type == REALSXP) {
    size_t seed_int = cpp11::as_cpp<size_t>(r_seed);
    seed = dust::random::seed_data<rng_state_type>(seed_int);
  } else if (seed_type == RAWSXP) {
    cpp11::raws seed_data = cpp11::as_cpp<cpp11::raws>(r_seed);
    constexpr size_t len = sizeof(int_type) * rng_state_type::size();
    if (seed_data.size() == 0 || seed_data.size() % len != 0) {
      cpp11::stop("Expected raw vector of length as multiple of %d for 'seed'",
                  len);
    }
    seed.resize(seed_data.size() / sizeof(int_type));
    std::memcpy(seed.data(), RAW(seed_data), seed_data.size());
  } else if (seed_type == NILSXP) {
    GetRNGstate();
    size_t seed_int =
      std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
    PutRNGstate();
    seed = dust::random::seed_data<rng_state_type>(seed_int);
  } else {
    cpp11::stop("Invalid type for 'seed'");
  }
  return seed;
}

template <typename rng_state_type>
SEXP rng_pointer_init(int n_streams, cpp11::sexp r_seed) {
  auto seed = as_rng_seed<rng_state_type>(r_seed);
  auto *rng = new prng<rng_state_type>(n_streams, seed);
  auto ret = cpp11::external_pointer<prng<rng_state_type>>(rng);
  return ret;
}

template <typename rng_state_type>
prng<rng_state_type>* rng_pointer_get(cpp11::sexp ptr, int n_streams = 0) {
  auto *rng =
    cpp11::as_cpp<cpp11::external_pointer<prng<rng_state_type>>>(ptr).get();
  if (n_streams > 0) {
    if (static_cast<int>(rng->size()) < n_streams) {
      cpp11::stop("Requested a rng with %d streams but only have %d",
                  n_streams, rng->size());
    }
  }
  return rng;
}

}
}
}

#endif
