// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"
#include <R_ext/Visibility.h>

// densities.cpp
SEXP dust_dbinom(cpp11::integers x, cpp11::integers size, cpp11::doubles prob, bool log);
extern "C" SEXP _dust_dust_dbinom(SEXP x, SEXP size, SEXP prob, SEXP log) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_dbinom(cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(x), cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(prob), cpp11::as_cpp<cpp11::decay_t<bool>>(log)));
  END_CPP11
}
// densities.cpp
SEXP dust_dnorm(cpp11::doubles x, cpp11::doubles mu, cpp11::doubles sd, bool log);
extern "C" SEXP _dust_dust_dnorm(SEXP x, SEXP mu, SEXP sd, SEXP log) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_dnorm(cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(x), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(mu), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(sd), cpp11::as_cpp<cpp11::decay_t<bool>>(log)));
  END_CPP11
}
// densities.cpp
SEXP dust_dnbinom_mu(cpp11::integers x, cpp11::doubles size, cpp11::doubles mu, bool log, bool is_float);
extern "C" SEXP _dust_dust_dnbinom_mu(SEXP x, SEXP size, SEXP mu, SEXP log, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_dnbinom_mu(cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(x), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(mu), cpp11::as_cpp<cpp11::decay_t<bool>>(log), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// densities.cpp
SEXP dust_dnbinom_prob(cpp11::integers x, cpp11::doubles size, cpp11::doubles prob, bool log);
extern "C" SEXP _dust_dust_dnbinom_prob(SEXP x, SEXP size, SEXP prob, SEXP log) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_dnbinom_prob(cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(x), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(prob), cpp11::as_cpp<cpp11::decay_t<bool>>(log)));
  END_CPP11
}
// densities.cpp
SEXP dust_dbetabinom(cpp11::integers x, cpp11::integers size, cpp11::doubles prob, cpp11::doubles rho, bool log);
extern "C" SEXP _dust_dust_dbetabinom(SEXP x, SEXP size, SEXP prob, SEXP rho, SEXP log) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_dbetabinom(cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(x), cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(prob), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(rho), cpp11::as_cpp<cpp11::decay_t<bool>>(log)));
  END_CPP11
}
// densities.cpp
SEXP dust_dpois(cpp11::integers x, cpp11::doubles lambda, bool log);
extern "C" SEXP _dust_dust_dpois(SEXP x, SEXP lambda, SEXP log) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_dpois(cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(x), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(lambda), cpp11::as_cpp<cpp11::decay_t<bool>>(log)));
  END_CPP11
}
// dust_rng.cpp
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators, bool is_float);
extern "C" SEXP _dust_dust_rng_alloc(SEXP r_seed, SEXP n_generators, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<int>>(n_generators), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
void dust_rng_jump(SEXP ptr, bool is_float);
extern "C" SEXP _dust_dust_rng_jump(SEXP ptr, SEXP is_float) {
  BEGIN_CPP11
    dust_rng_jump(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float));
    return R_NilValue;
  END_CPP11
}
// dust_rng.cpp
void dust_rng_long_jump(SEXP ptr, bool is_float);
extern "C" SEXP _dust_dust_rng_long_jump(SEXP ptr, SEXP is_float) {
  BEGIN_CPP11
    dust_rng_long_jump(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float));
    return R_NilValue;
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::doubles dust_rng_random_real(SEXP ptr, int n, bool is_float);
extern "C" SEXP _dust_dust_rng_random_real(SEXP ptr, SEXP n, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_random_real(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::doubles dust_rng_uniform(SEXP ptr, int n, cpp11::doubles r_min, cpp11::doubles r_max, bool is_float);
extern "C" SEXP _dust_dust_rng_uniform(SEXP ptr, SEXP n, SEXP r_min, SEXP r_max, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_uniform(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_min), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_max), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::doubles dust_rng_exponential(SEXP ptr, int n, cpp11::doubles r_rate, bool is_float);
extern "C" SEXP _dust_dust_rng_exponential(SEXP ptr, SEXP n, SEXP r_rate, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_exponential(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_rate), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::doubles dust_rng_normal(SEXP ptr, int n, cpp11::doubles r_mean, cpp11::doubles r_sd, bool is_float);
extern "C" SEXP _dust_dust_rng_normal(SEXP ptr, SEXP n, SEXP r_mean, SEXP r_sd, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_normal(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_mean), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_sd), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::doubles dust_rng_binomial(SEXP ptr, int n, cpp11::doubles r_size, cpp11::doubles r_prob, bool is_float);
extern "C" SEXP _dust_dust_rng_binomial(SEXP ptr, SEXP n, SEXP r_size, SEXP r_prob, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_binomial(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_prob), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::doubles dust_rng_poisson(SEXP ptr, int n, cpp11::doubles r_lambda, bool is_float);
extern "C" SEXP _dust_dust_rng_poisson(SEXP ptr, SEXP n, SEXP r_lambda, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_poisson(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_lambda), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::raws dust_rng_state(SEXP ptr, bool is_float);
extern "C" SEXP _dust_dust_rng_state(SEXP ptr, SEXP is_float) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(is_float)));
  END_CPP11
}
// openmp.cpp
cpp11::writable::list cpp_openmp_info();
extern "C" SEXP _dust_cpp_openmp_info() {
  BEGIN_CPP11
    return cpp11::as_sexp(cpp_openmp_info());
  END_CPP11
}
// test_rng.cpp
std::vector<std::string> test_xoshiro_run(std::string name);
extern "C" SEXP _dust_test_xoshiro_run(SEXP name) {
  BEGIN_CPP11
    return cpp11::as_sexp(test_xoshiro_run(cpp11::as_cpp<cpp11::decay_t<std::string>>(name)));
  END_CPP11
}
// tools.cpp
cpp11::list cpp_scale_log_weights(std::vector<double> w);
extern "C" SEXP _dust_cpp_scale_log_weights(SEXP w) {
  BEGIN_CPP11
    return cpp11::as_sexp(cpp_scale_log_weights(cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(w)));
  END_CPP11
}

extern "C" {
static const R_CallMethodDef CallEntries[] = {
    {"_dust_cpp_openmp_info",       (DL_FUNC) &_dust_cpp_openmp_info,       0},
    {"_dust_cpp_scale_log_weights", (DL_FUNC) &_dust_cpp_scale_log_weights, 1},
    {"_dust_dust_dbetabinom",       (DL_FUNC) &_dust_dust_dbetabinom,       5},
    {"_dust_dust_dbinom",           (DL_FUNC) &_dust_dust_dbinom,           4},
    {"_dust_dust_dnbinom_mu",       (DL_FUNC) &_dust_dust_dnbinom_mu,       5},
    {"_dust_dust_dnbinom_prob",     (DL_FUNC) &_dust_dust_dnbinom_prob,     4},
    {"_dust_dust_dnorm",            (DL_FUNC) &_dust_dust_dnorm,            4},
    {"_dust_dust_dpois",            (DL_FUNC) &_dust_dust_dpois,            3},
    {"_dust_dust_rng_alloc",        (DL_FUNC) &_dust_dust_rng_alloc,        3},
    {"_dust_dust_rng_binomial",     (DL_FUNC) &_dust_dust_rng_binomial,     5},
    {"_dust_dust_rng_exponential",  (DL_FUNC) &_dust_dust_rng_exponential,  4},
    {"_dust_dust_rng_jump",         (DL_FUNC) &_dust_dust_rng_jump,         2},
    {"_dust_dust_rng_long_jump",    (DL_FUNC) &_dust_dust_rng_long_jump,    2},
    {"_dust_dust_rng_normal",       (DL_FUNC) &_dust_dust_rng_normal,       5},
    {"_dust_dust_rng_poisson",      (DL_FUNC) &_dust_dust_rng_poisson,      4},
    {"_dust_dust_rng_random_real",  (DL_FUNC) &_dust_dust_rng_random_real,  3},
    {"_dust_dust_rng_state",        (DL_FUNC) &_dust_dust_rng_state,        2},
    {"_dust_dust_rng_uniform",      (DL_FUNC) &_dust_dust_rng_uniform,      5},
    {"_dust_test_xoshiro_run",      (DL_FUNC) &_dust_test_xoshiro_run,      1},
    {NULL, NULL, 0}
};
}

extern "C" attribute_visible void R_init_dust(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
