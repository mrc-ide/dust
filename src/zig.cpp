#include <cpp11/doubles.hpp>
#include "dust/random/random.hpp"
#include "dust/random/normal_ziggurat.hpp"

[[cpp11::register]]
cpp11::sexp normal_ziggurat(int n) {
  using rng_state_type = dust::random::xoroshiro128plus_state;
  auto state = dust::random::seed<rng_state_type>(42);
  cpp11::writable::doubles ret(n);
  const auto data = REAL(ret);
  for (int i = 0; i < n; ++i) {
    data[i] = dust::random::random_normal_ziggurat<double>(state);
  }
  return ret;
}

enum class normal_algorithm {box_muller, ziggurat};

template <typename real_type,
          typename rng_state_type>
real_type normal_general(rng_state_type& state,
                         normal_algorithm algorithm) {
  switch(algorithm) {
  case normal_algorithm::box_muller:
    return dust::random::random_normal_box_muller<real_type>(state);
  case normal_algorithm::ziggurat:
    return dust::random::random_normal_ziggurat<real_type, rng_state_type>(state);
  }
}

[[cpp11::register]]
cpp11::sexp normal2(int n, const bool use_ziggurat) {
  using rng_state_type = dust::random::xoroshiro128plus_state;
  auto state = dust::random::seed<rng_state_type>(42);
  const auto algorithm = use_ziggurat ?
    normal_algorithm::ziggurat : normal_algorithm::box_muller;
  cpp11::writable::doubles ret(n);
  const auto data = REAL(ret);
  for (int i = 0; i < n; ++i) {
    data[i] = normal_general<double>(state, algorithm);
  }
  return ret;
}

[[cpp11::register]]
cpp11::sexp normal3(int n) {
  using rng_state_type = dust::random::xoroshiro128plus_state;
  auto state = dust::random::seed<rng_state_type>(42);
  cpp11::writable::doubles ret(n);
  const auto data = REAL(ret);
  for (int i = 0; i < n; ++i) {
    data[i] = dust::random::random_normal<double, dust::random::algorithm::normal::ziggurat>(state);
  }
  return ret;
}
