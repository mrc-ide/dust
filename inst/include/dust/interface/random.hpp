#pragma once

#include <dust/random/generator.hpp>
#include <cpp11/raws.hpp>
#include <R_ext/Random.h>

namespace dust {
namespace interface {

template <typename rng_state_t>
std::vector<typename rng_state_t::data_type> as_rng_seed(cpp11::sexp r_seed) {
  typedef typename rng_state_t::data_type data_type;
  auto seed_type = TYPEOF(r_seed);
  std::vector<data_type> seed;
  if (seed_type == INTSXP || seed_type == REALSXP) {
    size_t seed_int = cpp11::as_cpp<size_t>(r_seed);
    seed = dust::random::seed_data<rng_state_t>(seed_int);
  } else if (seed_type == RAWSXP) {
    cpp11::raws seed_data = cpp11::as_cpp<cpp11::raws>(r_seed);
    const size_t len = sizeof(data_type) * rng_state_t::size();
    if (seed_data.size() == 0 || seed_data.size() % len != 0) {
      cpp11::stop("Expected raw vector of length as multiple of %d for 'seed'",
                  len);
    }
    seed.resize(seed_data.size() / sizeof(data_type));
    std::memcpy(seed.data(), RAW(seed_data), seed_data.size());
  } else if (seed_type == NILSXP) {
    GetRNGstate();
    size_t seed_int =
      std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
    PutRNGstate();
    seed = dust::random::seed_data<rng_state_t>(seed_int);
  } else {
    cpp11::stop("Invalid type for 'seed'");
  }
  return seed;
}

}
}
