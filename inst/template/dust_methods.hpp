/// IMPORTANT; changes here must be reflected in inst/template/dust_methods.cpp
[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                         cpp11::sexp r_n_particles, int n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config, cpp11::sexp ode_control);

[[cpp11::register]]
cpp11::sexp dust_{{target}}_{{name}}_capabilities();

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_run(SEXP ptr, cpp11::sexp r_time_end);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_run_adjoint(SEXP ptr);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_time(SEXP ptr);

[[cpp11::register]]
void dust_{{target}}_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_{{target}}_{{name}}_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_{{target}}_{{name}}_n_state(SEXP ptr);

[[cpp11::register]]
void dust_{{target}}_{{name}}_set_stochastic_schedule(SEXP ptr, SEXP time);

[[cpp11::register]]
SEXP dust_{{target}}_{{name}}_ode_statistics(SEXP ptr);
