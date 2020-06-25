#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <random>
#include <dust/xoshiro.hpp>
#include <dust/distr/binomial.hpp>
#include <dust/distr/poisson.hpp>

namespace dust {

template <typename float_t, typename int_t>
class RNG {
public:
  RNG(dust::Xoshiro<float_t> generator) : _generator(generator) {}

  float_t unif_rand() {
    return _generator.unif_rand();
  }

  float_t runif(float_t min, float_t max) {
    std::uniform_real_distribution<float_t> unif_dist(min, max);
    return unif_dist(_generator);
  }

  float_t rnorm(float_t mu, float_t sd) {
    std::normal_distribution<float_t> norm(mu, sd);
    return norm(_generator);
  }

  int_t rbinom(int_t n, float_t p) {
    return dust::distr::rbinom<float_t, int_t>(_generator, n, p);
  }

  int_t rpois(float_t lambda) {
    return dust::distr::rpois<float_t, int_t>(_generator, lambda);
  }

private:
  dust::Xoshiro<float_t> _generator;
};


template <typename float_t, typename int_t>
class pRNG {
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro<float_t> rng(seed);
    for (size_t i = 0; i < n; ++i) {
      _rngs.push_back(RNG<float_t, int_t>(rng));
      rng.jump();
    }
  }

  RNG<float_t, int_t>& operator()(size_t index) {
    return _rngs[index];
  }

  size_t size() const {
    return _rngs.size();
  }

private:
  std::vector<RNG<float_t, int_t>> _rngs;
};

}

#endif
