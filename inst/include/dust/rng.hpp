#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <random>
#include <dust/xoshiro.hpp>
#include <dust/distr/binomial.hpp>

namespace dust {

class RNG {
public:
  RNG(dust::Xoshiro generator) : _generator(generator) {}

  double unif_rand() {
    return _generator.unif_rand();
  }

  double runif(double min, double max) {
    std::uniform_real_distribution<double> unif_dist(min, max);
    return unif_dist(_generator);
  }

  double rnorm(double mu, double sd) {
    std::normal_distribution<double> norm(mu, sd);
    return norm(_generator);
  }

  template <class T = int> T rbinom(double p, int n) {
    return dust::distr::rbinom<T>(_generator, p, n);
  }

  template <class T = int> T rpois(double lambda);

private:
  dust::Xoshiro _generator;
};


class pRNG {
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro rng(seed);
    for (size_t i = 0; i < n; ++i) {
      _rngs.push_back(RNG(rng));
      rng.jump();
    }
  }

  RNG& operator()(size_t index) {
    return _rngs[index];
  }

  size_t size() const {
    return _rngs.size();
  }

private:
  std::vector<RNG> _rngs;
};

}

#endif
