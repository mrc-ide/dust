#include <dust/densities.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>

[[cpp11::register]]
SEXP dust_dbinom(cpp11::integers x, cpp11::integers size, cpp11::doubles prob,
                 bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::dbinom<double>(x[i], size[i], prob[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP dust_dnbinom(cpp11::integers x, cpp11::integers size, cpp11::doubles mu,
                  bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::dnbinom<double>(x[i], size[i], mu[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP dust_dbetabinom(cpp11::integers x, cpp11::integers size,
                     cpp11::doubles prob, cpp11::doubles rho, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::dbetabinom<double>(x[i], size[i], prob[i], rho[i], log);
  }
  return ret;
}

[[cpp11::register]]
SEXP dust_dpois(cpp11::integers x, cpp11::doubles lambda, bool log) {
  const size_t n = x.size();
  cpp11::writable::doubles ret(x.size());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dust::dpois<double>(x[i], lambda[i], log);
  }
  return ret;
}