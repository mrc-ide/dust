#ifndef DUST_R_RANDOM_HPP
#define DUST_R_RANDOM_HPP

#include <cstring> // memcpy

#include <cpp11/environment.hpp>
#include <cpp11/list.hpp>
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

namespace {

template<typename T>
std::string algorithm_name() {
  std::string ret;
  if (std::is_same<T, xoshiro128plus_state>::value) {
    ret = "xoshiro128plus";
  } else if (std::is_same<T, xoshiro128plusplus_state>::value) {
    ret = "xoshiro128plusplus";
  } else if (std::is_same<T, xoshiro128starstar_state>::value) {
    ret = "xoshiro128starstar";
  } else if (std::is_same<T, xoroshiro128plus_state>::value) {
    ret = "xoroshiro128plus";
  } else if (std::is_same<T, xoroshiro128plusplus_state>::value) {
    ret = "xoroshiro128plusplus";
  } else if (std::is_same<T, xoroshiro128starstar_state>::value) {
    ret = "xoroshiro128starstar";
  } else if (std::is_same<T, xoshiro256plus_state>::value) {
    ret = "xoshiro256plus";
  } else if (std::is_same<T, xoshiro256plusplus_state>::value) {
    ret = "xoshiro256plusplus";
  } else if (std::is_same<T, xoshiro256starstar_state>::value) {
    ret = "xoshiro256starstar";
  } else if (std::is_same<T, xoshiro512plus_state>::value) {
    ret = "xoshiro512plus";
  } else if (std::is_same<T, xoshiro512plusplus_state>::value) {
    ret = "xoshiro512plusplus";
  } else if (std::is_same<T, xoshiro512starstar_state>::value) {
    ret = "xoshiro512starstar";
  }
  return ret;
}

template <typename rng_state_type>
cpp11::raws rng_state_vector(prng<rng_state_type>* rng) {
  auto state = rng->export_state();
  size_t len = sizeof(typename rng_state_type::int_type) * state.size();
  cpp11::writable::raws r_state(len);
  std::memcpy(RAW(r_state), state.data(), len);
  return r_state;
}

}

template <typename rng_state_type>
SEXP rng_pointer_init(int n_streams, cpp11::sexp r_seed) {
  auto seed = as_rng_seed<rng_state_type>(r_seed);
  auto *rng = new prng<rng_state_type>(n_streams, seed);
  auto r_ptr = cpp11::external_pointer<prng<rng_state_type>>(rng);
  auto r_state = rng_state_vector(rng);
  return cpp11::writable::list({r_ptr, r_state});
}

// Start with the assumption that we'll pass in the R6 object, might
// write a simpler version later.
template <typename rng_state_type>
prng<rng_state_type>* rng_pointer_get(cpp11::environment obj,
                                      int n_streams = 0) {
  cpp11::environment env_enclos =
    cpp11::as_cpp<cpp11::environment>(obj[".__enclos_env__"]);
  cpp11::environment env =
    cpp11::as_cpp<cpp11::environment>(env_enclos["private"]);
  
  // We could probably do this more efficiently if we store an enum
  // in the object but this is probably ok.
  const auto algorithm_given = cpp11::as_cpp<std::string>(env["algorithm_"]);
  const auto algorithm_expected = algorithm_name<rng_state_type>();
  if (algorithm_given != algorithm_expected) {
    cpp11::stop("Incorrect rng type: given %s, expected %s",
                algorithm_given.c_str(), algorithm_expected.c_str());
  }

  using ptr_type = cpp11::external_pointer<prng<rng_state_type>>;
  auto ptr = cpp11::as_cpp<ptr_type>(env["ptr_"]);

  auto * rng = ptr.get();
  if (rng == nullptr) {
    if (!cpp11::as_cpp<bool>(env["is_current_"])) {
      cpp11::stop("Can't unserialise an rng pointer that was not synced");
    }

    using int_type = typename rng_state_type::int_type;
    cpp11::raws seed_data = cpp11::as_cpp<cpp11::raws>(env["state_"]);
    std::vector<int_type> seed(seed_data.size() / sizeof(int_type));
    std::memcpy(seed.data(), RAW(seed_data), seed_data.size());
    const auto n_streams_orig = seed.size() / rng_state_type::size();
    rng = new prng<rng_state_type>(n_streams_orig, seed);
    env["ptr_"] = cpp11::external_pointer<prng<rng_state_type>>(rng);
  }

  if (n_streams > 0 && static_cast<int>(rng->size()) < n_streams) {
    cpp11::stop("Requested a rng with %d streams but only have %d",
                n_streams, rng->size());
  }
  env["is_current_"] = cpp11::as_sexp(false);

  return rng;
}

template <typename rng_state_type>
void rng_pointer_sync(cpp11::environment obj) {
  using ptr_type = cpp11::external_pointer<prng<rng_state_type>>;
  auto ptr = cpp11::as_cpp<ptr_type>(obj["ptr_"]);
  obj["state_"] = rng_state_vector(ptr.get());
  obj["is_current_"] = cpp11::as_sexp(true);
}

}
}
}

#endif
