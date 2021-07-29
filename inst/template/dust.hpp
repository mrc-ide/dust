#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_{{name}}_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_config);

[[cpp11::register]]
SEXP dust_{{name}}_run(SEXP ptr, size_t step_end, bool device,
                       bool deterministic);

[[cpp11::register]]
SEXP dust_{{name}}_simulate(SEXP ptr, cpp11::sexp step_end, bool device,
                            bool deterministic);

[[cpp11::register]]
SEXP dust_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_{{name}}_set_state(SEXP ptr, SEXP r_state, SEXP r_step,
                             bool deterministic);

[[cpp11::register]]
SEXP dust_{{name}}_reset(SEXP ptr, cpp11::list r_pars, size_t step);

[[cpp11::register]]
SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_{{name}}_step(SEXP ptr);

[[cpp11::register]]
void dust_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_{{name}}_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_{{name}}_set_pars(SEXP ptr, cpp11::list r_pars);

[[cpp11::register]]
SEXP dust_{{name}}_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_{{name}}_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_{{name}}_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_{{name}}_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_{{name}}_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_{{name}}_capabilities();

[[cpp11::register]]
void dust_{{name}}_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_{{name}}_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_{{name}}_device_info();
