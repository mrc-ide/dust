#ifndef DUST_RANDOM_RANDOM_HPP
#define DUST_RANDOM_RANDOM_HPP

#include <dust/random/generator.hpp>
#include <dust/random/prng.hpp>

#include <dust/random/binomial.hpp>
#include <dust/random/exponential.hpp>
#include <dust/random/normal.hpp>
#include <dust/random/multinomial.hpp>
#include <dust/random/poisson.hpp>
#include <dust/random/uniform.hpp>

namespace dust {
namespace random {

namespace {

template <typename T>
struct default_rng_helper;

template <>
struct default_rng_helper<double> {
  typedef xoshiro256plus_state type;
};

template <>
struct default_rng_helper<float> {
  typedef xoshiro128plus_state type;
};

}

template <typename T>
using generator = typename default_rng_helper<T>::type;

}
}

#endif
