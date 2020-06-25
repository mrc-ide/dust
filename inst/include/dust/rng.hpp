#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <random>
#include <dust/xoshiro.hpp>
#include <dust/distr/binomial.hpp>
#include <dust/distr/poisson.hpp>

namespace dust {

template <typename FloatType = double, typename IntType = int>
class RNG {
public:
  RNG(dust::Xoshiro<FloatType> generator) : _generator(generator) {}

  FloatType unif_rand() {
    return _generator.unif_rand();
  }

  FloatType runif(FloatType min, FloatType max) {
    std::uniform_real_distribution<FloatType> unif_dist(min, max);
    return unif_dist(_generator);
  }

  FloatType rnorm(FloatType mu, FloatType sd) {
    std::normal_distribution<FloatType> norm(mu, sd);
    return norm(_generator);
  }

  IntType rbinom(IntType n, FloatType p) {
    return dust::distr::rbinom(_generator, n, p);
  }

  IntType rpois(FloatType lambda) {
    return dust::distr::rpois<IntType>(_generator, lambda);
  }

private:
  dust::Xoshiro<FloatType> _generator;
};


template <typename FloatType, typename IntType>
class pRNG {
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro<FloatType> rng(seed);
    for (size_t i = 0; i < n; ++i) {
      _rngs.push_back(RNG<FloatType, IntType>(rng));
      rng.jump();
    }
  }

  RNG<FloatType, IntType>& operator()(size_t index) {
    return _rngs[index];
  }

  size_t size() const {
    return _rngs.size();
  }

private:
  std::vector<RNG<FloatType, IntType>> _rngs;
};

}

#endif
