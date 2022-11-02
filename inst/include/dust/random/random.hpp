#ifndef DUST_RANDOM_RANDOM_HPP
#define DUST_RANDOM_RANDOM_HPP

#include "dust/random/generator.hpp"
#include "dust/random/prng.hpp"

#include "dust/random/binomial.hpp"
#include "dust/random/nbinomial.hpp"
#include "dust/random/exponential.hpp"
#include "dust/random/hypergeometric.hpp"
#include "dust/random/gamma.hpp"
#include "dust/random/multinomial.hpp"
#include "dust/random/normal.hpp"
#include "dust/random/poisson.hpp"
#include "dust/random/uniform.hpp"

#include "dust/random/version.hpp"

namespace dust {
namespace random {

namespace {

template <typename T>
struct default_rng_helper;

template <>
struct default_rng_helper<double> {
  using type = xoshiro256plus;
};

template <>
struct default_rng_helper<float> {
  using type = xoshiro128plus;
};

}

template <typename T>
using generator = typename default_rng_helper<T>::type;

}
}

#endif
