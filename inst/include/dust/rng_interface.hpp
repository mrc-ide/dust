#ifndef DUST_RNG_INTERFACE_HPP
#define DUST_RNG_INTERFACE_HPP

#include <R_ext/Random.h>

namespace dust {
namespace interface {

template <typename T>
std::vector<uint64_t> as_rng_seed(cpp11::sexp r_seed) {
  auto seed_type = TYPEOF(r_seed);
  std::vector<uint64_t> seed;
  if (seed_type == INTSXP || seed_type == REALSXP) {
    size_t seed_int = cpp11::as_cpp<size_t>(r_seed);
    seed = dust::xoshiro_initial_seed<T>(seed_int);
  } else if (seed_type == RAWSXP) {
    cpp11::raws seed_data = cpp11::as_cpp<cpp11::raws>(r_seed);
    const size_t len = sizeof(uint64_t) * dust::rng_state_t<T>::size();
    if (seed_data.size() == 0 || seed_data.size() % len != 0) {
      cpp11::stop("Expected raw vector of length as multiple of %d for 'seed'",
                  len);
    }
    seed.resize(seed_data.size() / sizeof(uint64_t));
    std::memcpy(seed.data(), RAW(seed_data), seed_data.size());
  } else if (seed_type == NILSXP) {
    GetRNGstate();
    size_t seed_int =
      std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
    PutRNGstate();
    seed = dust::xoshiro_initial_seed<T>(seed_int);
  } else {
    cpp11::stop("Invalid type for 'seed'");
  }
  return seed;
}

}
}

#endif
