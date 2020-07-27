#include <cpp11/list.hpp>

// These only exist so that cpp11 finds them as it can't look within
// .cu files
[[cpp11::register]]
SEXP dust_volatilitygpu_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                           size_t n_threads, size_t seed);

[[cpp11::register]]
SEXP dust_volatilitygpu_run(SEXP ptr, size_t step_end);

[[cpp11::register]]
SEXP dust_volatilitygpu_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_volatilitygpu_set_state(SEXP ptr, SEXP r_state, SEXP r_step);

[[cpp11::register]]
SEXP dust_volatilitygpu_reset(SEXP ptr, cpp11::list r_data, size_t step);

[[cpp11::register]]
SEXP dust_volatilitygpu_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_volatilitygpu_step(SEXP ptr);

[[cpp11::register]]
void dust_volatilitygpu_reorder(SEXP ptr, cpp11::sexp r_index);
