#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <random>
#include <dust/xoshiro.hpp>
#include <dust/distr/binomial.hpp>
#include <dust/distr/poisson.hpp>

namespace dust {

template <typename real_t, typename int_t>
class RNG {
public:
  RNG(dust::Xoshiro<real_t> generator) : _generator(generator) {}

  real_t unif_rand() {
    return _generator.unif_rand();
  }

  real_t norm_rand() {
    static std::normal_distribution<real_t> norm(0, 1);
    return norm(_generator);
  }

  real_t runif(real_t min, real_t max) {
    std::uniform_real_distribution<real_t> unif_dist(min, max);
    return unif_dist(_generator);
  }

  real_t rnorm(real_t mean, real_t sd) {
    std::normal_distribution<real_t> norm(mean, sd);
    return norm(_generator);
  }

  int_t rbinom(int_t size, real_t prob) {
    return dust::distr::rbinom<real_t, int_t>(_generator, size, prob);
  }

  int_t rpois(real_t lambda) {
    return dust::distr::rpois<real_t, int_t>(_generator, lambda);
  }

  void jump() {
    _generator.jump();
  }

  void long_jump() {
    _generator.long_jump();
  }

private:
  dust::Xoshiro<real_t> _generator;
};


template <typename real_t, typename int_t>
class pRNG {
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro<real_t> rng(seed);
    for (size_t i = 0; i < n; ++i) {
      _rngs.push_back(RNG<real_t, int_t>(rng));
      rng.jump();
    }
  }

  RNG<real_t, int_t>& operator()(size_t index) {
    return get(index);
  }

  RNG<real_t, int_t>& get(size_t index) {
    return _rngs[index];
  }

  size_t size() const {
    return _rngs.size();
  }

  void jump() {
    for (size_t i = 0; i < _rngs.size(); ++i) {
      _rngs[i].jump();
    }
  }

  void long_jump() {
    for (size_t i = 0; i < _rngs.size(); ++i) {
      _rngs[i].long_jump();
    }
  }

private:
  std::vector<RNG<real_t, int_t>> _rngs;
};

}

#endif
