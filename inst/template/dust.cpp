#include <dust/dust.hpp>
#include <dust/interface.hpp>

{{model}}

SEXP dust_{{name}}_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed) {
  return dust_alloc<{{class}}>(r_pars, pars_multi, step, r_n_particles,
                               n_threads, r_seed);
}

SEXP dust_{{name}}_run(SEXP ptr, size_t step_end, bool device) {
  return dust_run<{{class}}>(ptr, step_end, device);
}

SEXP dust_{{name}}_simulate(SEXP ptr, cpp11::sexp step_end) {
  return dust_simulate<{{class}}>(ptr, step_end);
}

SEXP dust_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<{{class}}>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_{{name}}_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<{{class}}>(ptr, r_state, r_step);
  return R_NilValue;
}

SEXP dust_{{name}}_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust_reset<{{class}}>(ptr, r_pars, step);
}

SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index) {
  return dust_state<{{class}}>(ptr, r_index);
}

size_t dust_{{name}}_step(SEXP ptr) {
  return dust_step<{{class}}>(ptr);
}

void dust_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<{{class}}>(ptr, r_index);
}

SEXP dust_{{name}}_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust_resample<{{class}}>(ptr, r_weights);
}

SEXP dust_{{name}}_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust_set_pars<{{class}}>(ptr, r_pars);
}

SEXP dust_{{name}}_rng_state(SEXP ptr, bool first_only) {
  return dust_rng_state<{{class}}>(ptr, first_only);
}

SEXP dust_{{name}}_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust_set_rng_state<{{class}}>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_{{name}}_set_data(SEXP ptr, cpp11::list data) {
  dust_set_data<{{class}}>(ptr, data);
  return R_NilValue;
}

SEXP dust_{{name}}_compare_data(SEXP ptr) {
  return dust_compare_data<{{class}}>(ptr);
}

SEXP dust_{{name}}_filter(SEXP ptr, bool save_history) {
  return dust_filter<{{class}}>(ptr, save_history);
}

cpp11::sexp dust_{{name}}_capabilities() {
  return dust_capabilities<{{name}}>();
}

void dust_{{name}}_set_n_threads(SEXP ptr, int n_threads) {
  return dust_set_n_threads<{{class}}>(ptr, n_threads);
}

int dust_{{name}}_n_state(SEXP ptr) {
  return dust_n_state<{{class}}>(ptr);
}
