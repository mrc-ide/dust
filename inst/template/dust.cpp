#include <iostream>
#include <dust/dust.hpp>
#include <dust/interface.hpp>

{{model}}

[[cpp11::register]]
SEXP dust_{{name}}_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                size_t n_threads, size_t seed) {
  return dust_alloc<{{type}}>(r_data, step, n_particles, n_threads, seed);
}

[[cpp11::register]]
SEXP dust_{{name}}_run(SEXP ptr, size_t step_end) {
  return dust_run<{{type}}>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<{{type}}>(ptr, r_index);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_{{name}}_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<{{type}}>(ptr, r_state, r_step);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_{{name}}_reset(SEXP ptr, cpp11::list r_data, size_t step) {
  return dust_reset<{{type}}>(ptr, r_data, step);
}

[[cpp11::register]]
SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index) {
  return dust_state<{{type}}>(ptr, r_index);
}

[[cpp11::register]]
size_t dust_{{name}}_step(SEXP ptr) {
  return dust_step<{{type}}>(ptr);
}

[[cpp11::register]]
void dust_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<{{type}}>(ptr, r_index);
}

[[cpp11::register]]
SEXP dust_{{name}}_rng_state(SEXP ptr) {
  return dust_rng_state<{{type}}>(ptr);
}

[[cpp11::register]]
SEXP dust_{{name}}_simulate(cpp11::list r_data,
                            cpp11::doubles_matrix r_state,
                            cpp11::sexp r_steps,
                            cpp11::sexp r_index,
                            const size_t n_threads,
                            const size_t seed) {
  return dust_simulate<{{type}}>(r_data, r_state, r_steps, r_index,
                                 n_threads, seed);
}
