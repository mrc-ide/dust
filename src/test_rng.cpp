#include <Rcpp.h>
#include <dust/rng.hpp>

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector test_rng_norm(int n, int seed, int n_generators) {
  const double mean = 0, sd = 1;

  dust::pRNG r(n_generators, seed);

  Rcpp::NumericVector y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = r(i % n_generators).rnorm(mean, sd);
  }

  return y;
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector test_rng_unif(int n, double min, double max, int seed,
                                  int n_generators) {
  bool std_unif =
    Rcpp::traits::is_na<REALSXP>(min) || Rcpp::traits::is_na<REALSXP>(max);

  dust::pRNG r(n_generators, seed);
  Rcpp::NumericVector y(n);

  for (int i = 0; i < n; ++i) {
    dust::RNG& rng = r(i % n_generators);
    if (std_unif) {
      y[i] = rng.unif_rand();
    } else {
      y[i] = rng.runif(min, max);
    }
  }

  return y;
}

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerVector test_rng_binom(std::vector<int> n, std::vector<double> p,
                                   int seed, int n_generators) {
  size_t n_samples = n.size();
  dust::pRNG r(n_generators, seed);

  Rcpp::IntegerVector y(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    y[i] = r(i % n_generators).rbinom(n[i], p[i]);
  }

  return y;
}

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerVector test_rng_pois(std::vector<double> lambda,
                                  int seed, int n_generators) {
  size_t n_samples = lambda.size();
  dust::pRNG r(n_generators, seed);

  Rcpp::IntegerVector y(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    y[i] = r(i % n_generators).rpois(lambda[i]);
  }

  return y;
}
