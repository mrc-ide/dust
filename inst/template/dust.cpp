#include <iostream>
#include <dust/dust.hpp>
#include <dust/interface.hpp>

{{model}}

[[cpp11::register]]
SEXP dust_{{name}}_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         size_t n_particles, size_t n_threads,
                         cpp11::sexp r_seed) {
  return dust_alloc<{{class}}>(r_pars, pars_multi, step, n_particles,
                               n_threads, r_seed);
}

[[cpp11::register]]
SEXP dust_{{name}}_run(SEXP ptr, size_t step_end) {
  return dust_run<{{class}}>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<{{class}}>(ptr, r_index);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_{{name}}_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<{{class}}>(ptr, r_state, r_step);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_{{name}}_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust_reset<{{class}}>(ptr, r_pars, step);
}

[[cpp11::register]]
SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index) {
  return dust_state<{{class}}>(ptr, r_index);
}

[[cpp11::register]]
size_t dust_{{name}}_step(SEXP ptr) {
  return dust_step<{{class}}>(ptr);
}

[[cpp11::register]]
void dust_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<{{class}}>(ptr, r_index);
}

[[cpp11::register]]
SEXP dust_{{name}}_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust_set_pars<{{class}}>(ptr, r_pars);
}

[[cpp11::register]]
SEXP dust_{{name}}_rng_state(SEXP ptr, bool first_only) {
  return dust_rng_state<{{class}}>(ptr, first_only);
}

[[cpp11::register]]
SEXP dust_{{name}}_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust_set_rng_state<{{class}}>(ptr, rng_state);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_{{name}}_set_data(SEXP ptr, cpp11::list data) {
  dust_set_data<{{class}}>(ptr, data);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_{{name}}_compare_data(SEXP ptr) {
  return dust_compare_data<{{class}}>(ptr);
}

[[cpp11::register]]
SEXP dust_{{name}}_simulate(cpp11::sexp r_steps,
                            cpp11::list r_pars,
                            cpp11::doubles_matrix r_state,
                            cpp11::sexp r_index,
                            const size_t n_threads,
                            cpp11::sexp r_seed,
                            bool return_state) {
  return dust_simulate<{{class}}>(r_steps, r_pars, r_state, r_index,
                                  n_threads, r_seed, return_state);
}

[[cpp11::register]]
bool dust_{{name}}_has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

[[cpp11::register]]
void dust_{{name}}_set_n_threads(SEXP ptr, int n_threads) {
  return dust_set_n_threads<{{class}}>(ptr, n_threads);
}
