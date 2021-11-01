/// IMPORTANT; changes here must be reflected in inst/template/dust_methods.hpp
using model_{{target}} = dust::{{container}}<{{class}}>;

SEXP dust_{{target}}_{{name}}_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                             cpp11::sexp r_n_particles, size_t n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp device_config) {
  return dust::r::dust_{{target}}_alloc<{{class}}>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        device_config);
}

SEXP dust_{{target}}_{{name}}_run(SEXP ptr, size_t step_end) {
  return dust::r::dust_run<model_{{target}}>(ptr, step_end);
}

SEXP dust_{{target}}_{{name}}_simulate(SEXP ptr, cpp11::sexp step_end) {
  return dust::r::dust_simulate<model_{{target}}>(ptr, step_end);
}

SEXP dust_{{target}}_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_{{target}}>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_{{target}}_{{name}}_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state) {
  return dust::r::dust_update_state<model_{{target}}>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state);
}

SEXP dust_{{target}}_{{name}}_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_{{target}}>(ptr, r_index);
}

size_t dust_{{target}}_{{name}}_step(SEXP ptr) {
  return dust::r::dust_step<model_{{target}}>(ptr);
}

void dust_{{target}}_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_{{target}}>(ptr, r_index);
}

SEXP dust_{{target}}_{{name}}_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_{{target}}>(ptr, r_weights);
}

SEXP dust_{{target}}_{{name}}_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_{{target}}>(ptr, first_only, last_only);
}

SEXP dust_{{target}}_{{name}}_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_{{target}}>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_{{target}}_{{name}}_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<model_{{target}}>(ptr, data);
  return R_NilValue;
}

SEXP dust_{{target}}_{{name}}_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_{{target}}>(ptr);
}

SEXP dust_{{target}}_{{name}}_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot) {
  return dust::r::dust_filter<model_{{target}}>(ptr, save_trajectories, step_snapshot);
}

void dust_{{target}}_{{name}}_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_{{target}}>(ptr, n_threads);
}

int dust_{{target}}_{{name}}_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_{{target}}>(ptr);
}