#include <Rcpp.h>
#include <dust/rng.hpp>

typedef dust::pRNG<double, int> dust_rng_t;

// [[Rcpp::export(rng = false)]]
SEXP dust_rng_alloc(int seed, int n_generators) {
  dust_rng_t *rng = new dust_rng_t(n_generators, seed);
  Rcpp::XPtr<dust_rng_t> ptr(rng, false);
  return ptr;
}

// [[Rcpp::export(rng = false)]]
std::vector<double> dust_rng_unif_rand(SEXP ptr, int n) {
  dust_rng_t *rng = Rcpp::as<Rcpp::XPtr<dust_rng_t>>(ptr);
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = rng->get(i % n_generators).unif_rand();
  }
  return y;
}

// [[Rcpp::export(rng = false)]]
std::vector<double> dust_rng_norm_rand(SEXP ptr, int n) {
  dust_rng_t *rng = Rcpp::as<Rcpp::XPtr<dust_rng_t>>(ptr);
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = rng->get(i % n_generators).norm_rand();
  }
  return y;
}

// [[Rcpp::export(rng = false)]]
std::vector<double> dust_rng_runif(SEXP ptr, int n, std::vector<double> min,
                                   std::vector<double> max) {
  dust_rng_t *rng = Rcpp::as<Rcpp::XPtr<dust_rng_t>>(ptr);
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = rng->get(i % n_generators).runif(min[i], max[i]);
  }
  return y;
}

// [[Rcpp::export(rng = false)]]
std::vector<double> dust_rng_rnorm(SEXP ptr, int n, std::vector<double> mean,
                                   std::vector<double> sd) {
  dust_rng_t *rng = Rcpp::as<Rcpp::XPtr<dust_rng_t>>(ptr);
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = rng->get(i % n_generators).rnorm(mean[i], sd[i]);
  }
  return y;
}

// [[Rcpp::export(rng = false)]]
std::vector<int> dust_rng_rbinom(SEXP ptr, int n, std::vector<int> size,
                                 std::vector<double> prob) {
  dust_rng_t *rng = Rcpp::as<Rcpp::XPtr<dust_rng_t>>(ptr);
  const size_t n_generators = rng->size();
  std::vector<int> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = rng->get(i % n_generators).rbinom(size[i], prob[i]);
  }
  return y;
}

// [[Rcpp::export(rng = false)]]
std::vector<int>  dust_rng_rpois(SEXP ptr, int n,
                                    std::vector<double> lambda) {
  dust_rng_t *rng = Rcpp::as<Rcpp::XPtr<dust_rng_t>>(ptr);
  const size_t n_generators = rng->size();
  std::vector<int> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = rng->get(i % n_generators).rpois(lambda[i]);
  }
  return y;
}
