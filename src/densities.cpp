#include <dust/random/density.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>

[[cpp11::register]]
SEXP density_binomial(cpp11::integers x, cpp11::integers size,
                      cpp11::doubles prob, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::density::binomial<double>(x[i], size[i], prob[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP density_normal(cpp11::doubles x, cpp11::doubles mu, cpp11::doubles sd,
                    bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::density::normal<double>(x[i], mu[i], sd[i], log);
  }
  return ret;
}

template <typename T>
SEXP density_negative_binomial_mu_(cpp11::integers x, cpp11::doubles size,
                                   cpp11::doubles mu, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::density::negative_binomial_mu<T>(x[i], size[i], mu[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP density_negative_binomial_mu(cpp11::integers x, cpp11::doubles size,
                                  cpp11::doubles mu, bool log, bool is_float) {
  return is_float ?
    density_negative_binomial_mu_<float>(x, size, mu, log) :
    density_negative_binomial_mu_<double>(x, size, mu, log);
}

[[cpp11::register]]
SEXP density_negative_binomial_prob(cpp11::integers x, cpp11::doubles size,
                                    cpp11::doubles prob, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::density::negative_binomial_prob<double>(x[i], size[i],
                                                           prob[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP density_beta_binomial(cpp11::integers x, cpp11::integers size,
                           cpp11::doubles prob, cpp11::doubles rho, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::density::beta_binomial<double>(x[i], size[i], prob[i],
                                                  rho[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP density_poisson(cpp11::integers x, cpp11::doubles lambda, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::density::poisson<double>(x[i], lambda[i], log);
  }
  return ret;
}
