// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"

// dust_rng.cpp
SEXP dust_rng_alloc(cpp11::sexp r_seed, int n_generators);
extern "C" SEXP _dust_dust_rng_alloc(SEXP r_seed, SEXP n_generators) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<int>>(n_generators)));
  END_CPP11
}
// dust_rng.cpp
int dust_rng_size(SEXP ptr);
extern "C" SEXP _dust_dust_rng_size(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_size(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// dust_rng.cpp
void dust_rng_jump(SEXP ptr);
extern "C" SEXP _dust_dust_rng_jump(SEXP ptr) {
  BEGIN_CPP11
    dust_rng_jump(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr));
    return R_NilValue;
  END_CPP11
}
// dust_rng.cpp
void dust_rng_long_jump(SEXP ptr);
extern "C" SEXP _dust_dust_rng_long_jump(SEXP ptr) {
  BEGIN_CPP11
    dust_rng_long_jump(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr));
    return R_NilValue;
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_unif_rand(SEXP ptr, int n);
extern "C" SEXP _dust_dust_rng_unif_rand(SEXP ptr, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_unif_rand(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n)));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_norm_rand(SEXP ptr, int n);
extern "C" SEXP _dust_dust_rng_norm_rand(SEXP ptr, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_norm_rand(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n)));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_runif(SEXP ptr, int n, std::vector<double> min, std::vector<double> max);
extern "C" SEXP _dust_dust_rng_runif(SEXP ptr, SEXP n, SEXP min, SEXP max) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_runif(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(min), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(max)));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_rnorm(SEXP ptr, int n, std::vector<double> mean, std::vector<double> sd);
extern "C" SEXP _dust_dust_rng_rnorm(SEXP ptr, SEXP n, SEXP mean, SEXP sd) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_rnorm(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(mean), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(sd)));
  END_CPP11
}
// dust_rng.cpp
std::vector<int> dust_rng_rbinom(SEXP ptr, int n, std::vector<int> size, std::vector<double> prob);
extern "C" SEXP _dust_dust_rng_rbinom(SEXP ptr, SEXP n, SEXP size, SEXP prob) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_rbinom(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<int>>>(size), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(prob)));
  END_CPP11
}
// dust_rng.cpp
std::vector<int> dust_rng_rpois(SEXP ptr, int n, std::vector<double> lambda);
extern "C" SEXP _dust_dust_rng_rpois(SEXP ptr, SEXP n, SEXP lambda) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_rpois(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(lambda)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::raws dust_rng_state(SEXP ptr);
extern "C" SEXP _dust_dust_rng_state(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// openmp.cpp
cpp11::writable::list cpp_openmp_info();
extern "C" SEXP _dust_cpp_openmp_info() {
  BEGIN_CPP11
    return cpp11::as_sexp(cpp_openmp_info());
  END_CPP11
}
// sir.cpp
SEXP dust_sir_alloc(cpp11::list r_data, size_t step, size_t n_particles, size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_sir_alloc(SEXP r_data, SEXP step, SEXP n_particles, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_particles), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_run(SEXP ptr, size_t step_end);
extern "C" SEXP _dust_dust_sir_run(SEXP ptr, SEXP step_end) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<size_t>>(step_end)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_set_index(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_sir_set_index(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step);
extern "C" SEXP _dust_dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_set_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_step)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_reset(SEXP ptr, cpp11::list r_data, size_t step);
extern "C" SEXP _dust_dust_sir_reset(SEXP ptr, SEXP r_data, SEXP step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_reset(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_state(SEXP ptr, SEXP r_index);
extern "C" SEXP _dust_dust_sir_state(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
  END_CPP11
}
// sir.cpp
size_t dust_sir_step(SEXP ptr);
extern "C" SEXP _dust_dust_sir_step(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_step(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// sir.cpp
void dust_sir_reorder(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_sir_reorder(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    dust_sir_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
    return R_NilValue;
  END_CPP11
}
// sir.cpp
SEXP dust_sir_rng_state(SEXP ptr, bool advance);
extern "C" SEXP _dust_dust_sir_rng_state(SEXP ptr, SEXP advance) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(advance)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_simulate(cpp11::sexp r_steps, cpp11::list r_data, cpp11::doubles_matrix r_state, cpp11::sexp r_index, const size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_sir_simulate(SEXP r_steps, SEXP r_data, SEXP r_state, SEXP r_index, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_simulate(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_steps), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix>>(r_state), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index), cpp11::as_cpp<cpp11::decay_t<const size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_alloc(cpp11::list r_data, size_t step, size_t n_particles, size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_variable_alloc(SEXP r_data, SEXP step, SEXP n_particles, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_particles), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_run(SEXP ptr, size_t step_end);
extern "C" SEXP _dust_dust_variable_run(SEXP ptr, SEXP step_end) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<size_t>>(step_end)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_set_index(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_variable_set_index(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_set_state(SEXP ptr, SEXP r_state, SEXP r_step);
extern "C" SEXP _dust_dust_variable_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_set_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_step)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_reset(SEXP ptr, cpp11::list r_data, size_t step);
extern "C" SEXP _dust_dust_variable_reset(SEXP ptr, SEXP r_data, SEXP step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_reset(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_state(SEXP ptr, SEXP r_index);
extern "C" SEXP _dust_dust_variable_state(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
  END_CPP11
}
// variable.cpp
size_t dust_variable_step(SEXP ptr);
extern "C" SEXP _dust_dust_variable_step(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_step(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// variable.cpp
void dust_variable_reorder(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_variable_reorder(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    dust_variable_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
    return R_NilValue;
  END_CPP11
}
// variable.cpp
SEXP dust_variable_rng_state(SEXP ptr, bool advance);
extern "C" SEXP _dust_dust_variable_rng_state(SEXP ptr, SEXP advance) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(advance)));
  END_CPP11
}
// variable.cpp
SEXP dust_variable_simulate(cpp11::sexp r_steps, cpp11::list r_data, cpp11::doubles_matrix r_state, cpp11::sexp r_index, const size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_variable_simulate(SEXP r_steps, SEXP r_data, SEXP r_state, SEXP r_index, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_variable_simulate(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_steps), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix>>(r_state), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index), cpp11::as_cpp<cpp11::decay_t<const size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_alloc(cpp11::list r_data, size_t step, size_t n_particles, size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_volatility_alloc(SEXP r_data, SEXP step, SEXP n_particles, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_particles), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_run(SEXP ptr, size_t step_end);
extern "C" SEXP _dust_dust_volatility_run(SEXP ptr, SEXP step_end) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<size_t>>(step_end)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_set_index(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_volatility_set_index(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_set_state(SEXP ptr, SEXP r_state, SEXP r_step);
extern "C" SEXP _dust_dust_volatility_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_set_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_step)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_reset(SEXP ptr, cpp11::list r_data, size_t step);
extern "C" SEXP _dust_dust_volatility_reset(SEXP ptr, SEXP r_data, SEXP step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_reset(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_state(SEXP ptr, SEXP r_index);
extern "C" SEXP _dust_dust_volatility_state(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
  END_CPP11
}
// volatility.cpp
size_t dust_volatility_step(SEXP ptr);
extern "C" SEXP _dust_dust_volatility_step(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_step(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// volatility.cpp
void dust_volatility_reorder(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_volatility_reorder(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    dust_volatility_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
    return R_NilValue;
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_rng_state(SEXP ptr, bool advance);
extern "C" SEXP _dust_dust_volatility_rng_state(SEXP ptr, SEXP advance) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(advance)));
  END_CPP11
}
// volatility.cpp
SEXP dust_volatility_simulate(cpp11::sexp r_steps, cpp11::list r_data, cpp11::doubles_matrix r_state, cpp11::sexp r_index, const size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_volatility_simulate(SEXP r_steps, SEXP r_data, SEXP r_state, SEXP r_index, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_simulate(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_steps), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix>>(r_state), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index), cpp11::as_cpp<cpp11::decay_t<const size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_alloc(cpp11::list r_data, size_t step, size_t n_particles, size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_walk_alloc(SEXP r_data, SEXP step, SEXP n_particles, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_particles), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_run(SEXP ptr, size_t step_end);
extern "C" SEXP _dust_dust_walk_run(SEXP ptr, SEXP step_end) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<size_t>>(step_end)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_set_index(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_walk_set_index(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_set_state(SEXP ptr, SEXP r_state, SEXP r_step);
extern "C" SEXP _dust_dust_walk_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_set_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_step)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_reset(SEXP ptr, cpp11::list r_data, size_t step);
extern "C" SEXP _dust_dust_walk_reset(SEXP ptr, SEXP r_data, SEXP step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_reset(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_state(SEXP ptr, SEXP r_index);
extern "C" SEXP _dust_dust_walk_state(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
  END_CPP11
}
// walk.cpp
size_t dust_walk_step(SEXP ptr);
extern "C" SEXP _dust_dust_walk_step(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_step(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// walk.cpp
void dust_walk_reorder(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dust_dust_walk_reorder(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    dust_walk_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
    return R_NilValue;
  END_CPP11
}
// walk.cpp
SEXP dust_walk_rng_state(SEXP ptr, bool advance);
extern "C" SEXP _dust_dust_walk_rng_state(SEXP ptr, SEXP advance) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(advance)));
  END_CPP11
}
// walk.cpp
SEXP dust_walk_simulate(cpp11::sexp r_steps, cpp11::list r_data, cpp11::doubles_matrix r_state, cpp11::sexp r_index, const size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dust_dust_walk_simulate(SEXP r_steps, SEXP r_data, SEXP r_state, SEXP r_index, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_walk_simulate(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_steps), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix>>(r_state), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index), cpp11::as_cpp<cpp11::decay_t<const size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}

extern "C" {
/* .Call calls */
extern SEXP _dust_cpp_openmp_info();
extern SEXP _dust_dust_rng_alloc(SEXP, SEXP);
extern SEXP _dust_dust_rng_jump(SEXP);
extern SEXP _dust_dust_rng_long_jump(SEXP);
extern SEXP _dust_dust_rng_norm_rand(SEXP, SEXP);
extern SEXP _dust_dust_rng_rbinom(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_rnorm(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_rpois(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_runif(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_size(SEXP);
extern SEXP _dust_dust_rng_state(SEXP);
extern SEXP _dust_dust_rng_unif_rand(SEXP, SEXP);
extern SEXP _dust_dust_sir_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_sir_reorder(SEXP, SEXP);
extern SEXP _dust_dust_sir_reset(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_sir_rng_state(SEXP, SEXP);
extern SEXP _dust_dust_sir_run(SEXP, SEXP);
extern SEXP _dust_dust_sir_set_index(SEXP, SEXP);
extern SEXP _dust_dust_sir_set_state(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_sir_simulate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_sir_state(SEXP, SEXP);
extern SEXP _dust_dust_sir_step(SEXP);
extern SEXP _dust_dust_variable_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_variable_reorder(SEXP, SEXP);
extern SEXP _dust_dust_variable_reset(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_variable_rng_state(SEXP, SEXP);
extern SEXP _dust_dust_variable_run(SEXP, SEXP);
extern SEXP _dust_dust_variable_set_index(SEXP, SEXP);
extern SEXP _dust_dust_variable_set_state(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_variable_simulate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_variable_state(SEXP, SEXP);
extern SEXP _dust_dust_variable_step(SEXP);
extern SEXP _dust_dust_volatility_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_volatility_reorder(SEXP, SEXP);
extern SEXP _dust_dust_volatility_reset(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_volatility_rng_state(SEXP, SEXP);
extern SEXP _dust_dust_volatility_run(SEXP, SEXP);
extern SEXP _dust_dust_volatility_set_index(SEXP, SEXP);
extern SEXP _dust_dust_volatility_set_state(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_volatility_simulate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_volatility_state(SEXP, SEXP);
extern SEXP _dust_dust_volatility_step(SEXP);
extern SEXP _dust_dust_walk_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_walk_reorder(SEXP, SEXP);
extern SEXP _dust_dust_walk_reset(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_walk_rng_state(SEXP, SEXP);
extern SEXP _dust_dust_walk_run(SEXP, SEXP);
extern SEXP _dust_dust_walk_set_index(SEXP, SEXP);
extern SEXP _dust_dust_walk_set_state(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_walk_simulate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_walk_state(SEXP, SEXP);
extern SEXP _dust_dust_walk_step(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_dust_cpp_openmp_info",           (DL_FUNC) &_dust_cpp_openmp_info,           0},
    {"_dust_dust_rng_alloc",            (DL_FUNC) &_dust_dust_rng_alloc,            2},
    {"_dust_dust_rng_jump",             (DL_FUNC) &_dust_dust_rng_jump,             1},
    {"_dust_dust_rng_long_jump",        (DL_FUNC) &_dust_dust_rng_long_jump,        1},
    {"_dust_dust_rng_norm_rand",        (DL_FUNC) &_dust_dust_rng_norm_rand,        2},
    {"_dust_dust_rng_rbinom",           (DL_FUNC) &_dust_dust_rng_rbinom,           4},
    {"_dust_dust_rng_rnorm",            (DL_FUNC) &_dust_dust_rng_rnorm,            4},
    {"_dust_dust_rng_rpois",            (DL_FUNC) &_dust_dust_rng_rpois,            3},
    {"_dust_dust_rng_runif",            (DL_FUNC) &_dust_dust_rng_runif,            4},
    {"_dust_dust_rng_size",             (DL_FUNC) &_dust_dust_rng_size,             1},
    {"_dust_dust_rng_state",            (DL_FUNC) &_dust_dust_rng_state,            1},
    {"_dust_dust_rng_unif_rand",        (DL_FUNC) &_dust_dust_rng_unif_rand,        2},
    {"_dust_dust_sir_alloc",            (DL_FUNC) &_dust_dust_sir_alloc,            5},
    {"_dust_dust_sir_reorder",          (DL_FUNC) &_dust_dust_sir_reorder,          2},
    {"_dust_dust_sir_reset",            (DL_FUNC) &_dust_dust_sir_reset,            3},
    {"_dust_dust_sir_rng_state",        (DL_FUNC) &_dust_dust_sir_rng_state,        2},
    {"_dust_dust_sir_run",              (DL_FUNC) &_dust_dust_sir_run,              2},
    {"_dust_dust_sir_set_index",        (DL_FUNC) &_dust_dust_sir_set_index,        2},
    {"_dust_dust_sir_set_state",        (DL_FUNC) &_dust_dust_sir_set_state,        3},
    {"_dust_dust_sir_simulate",         (DL_FUNC) &_dust_dust_sir_simulate,         6},
    {"_dust_dust_sir_state",            (DL_FUNC) &_dust_dust_sir_state,            2},
    {"_dust_dust_sir_step",             (DL_FUNC) &_dust_dust_sir_step,             1},
    {"_dust_dust_variable_alloc",       (DL_FUNC) &_dust_dust_variable_alloc,       5},
    {"_dust_dust_variable_reorder",     (DL_FUNC) &_dust_dust_variable_reorder,     2},
    {"_dust_dust_variable_reset",       (DL_FUNC) &_dust_dust_variable_reset,       3},
    {"_dust_dust_variable_rng_state",   (DL_FUNC) &_dust_dust_variable_rng_state,   2},
    {"_dust_dust_variable_run",         (DL_FUNC) &_dust_dust_variable_run,         2},
    {"_dust_dust_variable_set_index",   (DL_FUNC) &_dust_dust_variable_set_index,   2},
    {"_dust_dust_variable_set_state",   (DL_FUNC) &_dust_dust_variable_set_state,   3},
    {"_dust_dust_variable_simulate",    (DL_FUNC) &_dust_dust_variable_simulate,    6},
    {"_dust_dust_variable_state",       (DL_FUNC) &_dust_dust_variable_state,       2},
    {"_dust_dust_variable_step",        (DL_FUNC) &_dust_dust_variable_step,        1},
    {"_dust_dust_volatility_alloc",     (DL_FUNC) &_dust_dust_volatility_alloc,     5},
    {"_dust_dust_volatility_reorder",   (DL_FUNC) &_dust_dust_volatility_reorder,   2},
    {"_dust_dust_volatility_reset",     (DL_FUNC) &_dust_dust_volatility_reset,     3},
    {"_dust_dust_volatility_rng_state", (DL_FUNC) &_dust_dust_volatility_rng_state, 2},
    {"_dust_dust_volatility_run",       (DL_FUNC) &_dust_dust_volatility_run,       2},
    {"_dust_dust_volatility_set_index", (DL_FUNC) &_dust_dust_volatility_set_index, 2},
    {"_dust_dust_volatility_set_state", (DL_FUNC) &_dust_dust_volatility_set_state, 3},
    {"_dust_dust_volatility_simulate",  (DL_FUNC) &_dust_dust_volatility_simulate,  6},
    {"_dust_dust_volatility_state",     (DL_FUNC) &_dust_dust_volatility_state,     2},
    {"_dust_dust_volatility_step",      (DL_FUNC) &_dust_dust_volatility_step,      1},
    {"_dust_dust_walk_alloc",           (DL_FUNC) &_dust_dust_walk_alloc,           5},
    {"_dust_dust_walk_reorder",         (DL_FUNC) &_dust_dust_walk_reorder,         2},
    {"_dust_dust_walk_reset",           (DL_FUNC) &_dust_dust_walk_reset,           3},
    {"_dust_dust_walk_rng_state",       (DL_FUNC) &_dust_dust_walk_rng_state,       2},
    {"_dust_dust_walk_run",             (DL_FUNC) &_dust_dust_walk_run,             2},
    {"_dust_dust_walk_set_index",       (DL_FUNC) &_dust_dust_walk_set_index,       2},
    {"_dust_dust_walk_set_state",       (DL_FUNC) &_dust_dust_walk_set_state,       3},
    {"_dust_dust_walk_simulate",        (DL_FUNC) &_dust_dust_walk_simulate,        6},
    {"_dust_dust_walk_state",           (DL_FUNC) &_dust_dust_walk_state,           2},
    {"_dust_dust_walk_step",            (DL_FUNC) &_dust_dust_walk_step,            1},
    {NULL, NULL, 0}
};
}

extern "C" void R_init_dust(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
