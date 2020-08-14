#include <cpp11/list.hpp>

// These only exist so that cpp11 finds them as it can't look within
// .cu files
[[cpp11::register]]
SEXP dust_{{name}}_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                           size_t n_threads, cpp11::sexp seed);

[[cpp11::register]]
SEXP dust_{{name}}_run(SEXP ptr, size_t step_end);

[[cpp11::register]]
SEXP dust_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_{{name}}_set_state(SEXP ptr, SEXP r_state, SEXP r_step);

[[cpp11::register]]
SEXP dust_{{name}}_reset(SEXP ptr, cpp11::list r_data, size_t step);

[[cpp11::register]]
SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_{{name}}_step(SEXP ptr);

[[cpp11::register]]
void dust_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index);
