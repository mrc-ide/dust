#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <random>
#include <dust/xoshiro.hpp>

namespace dust {

class RNG2 {
public:
  RNG2(dust::Xoshiro generator) : _generator(generator) {}

  double runif(double min, double max) {
    static std::uniform_real_distribution<double> unif_dist(min, max);
    return(unif_dist(_generator));
  }
  double rnorm(double mu, double sd) {
    std::normal_distribution<double> norm(mu, sd);
    return(norm(_generator));
  }
  template <class T = int> T rbinom(double p, int n);
  template <class T = int> T rpois(double lambda);

private:
  dust::Xoshiro _generator;
};


class pRNG {
public:
  pRNG(const size_t n, const uint64_t seed) {
    dust::Xoshiro rng(seed);
    for (size_t i = 0; i < n; ++i) {
      _rngs.push_back(RNG2(rng));
      rng.jump();
    }
  }

  RNG2& operator()(size_t index) {
    return _rngs[index];
  }

  size_t size() const {
    return _rngs.size();
  }

private:
  std::vector<RNG2> _rngs;
};

}

#endif
