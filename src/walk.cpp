// Generated by dust (version 0.10.0) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_walk_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp device_config);

[[cpp11::register]]
SEXP dust_walk_run(SEXP ptr, size_t step_end, bool device);

[[cpp11::register]]
SEXP dust_walk_simulate(SEXP ptr, cpp11::sexp step_end, bool device);

[[cpp11::register]]
SEXP dust_walk_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_walk_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state);

[[cpp11::register]]
SEXP dust_walk_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_walk_step(SEXP ptr);

[[cpp11::register]]
void dust_walk_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_walk_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_walk_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_walk_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_walk_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_walk_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_walk_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_walk_capabilities();

[[cpp11::register]]
void dust_walk_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_walk_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_walk_device_info();

#include <dust/dust.hpp>
#include <dust/interface/dust.hpp>

class walk {
public:
  typedef double real_type;
  typedef dust::no_data data_type;
  typedef dust::no_internal internal_type;
  typedef dust::random::generator<real_type> rng_state_type;

  struct shared_type {
    real_type sd;
  };

  walk(const dust::pars_type<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_type> initial(size_t step) {
    std::vector<real_type> ret = {0};
    return ret;
  }

  void update(size_t step, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type mean = state[0];
    state_next[0] = dust::random::normal<real_type>(rng_state, mean, shared->sd);
  }

private:
  dust::shared_ptr<walk> shared;
};

namespace dust {

template <>
dust::pars_type<walk> dust_pars<walk>(cpp11::list pars) {
  walk::real_type sd = cpp11::as_cpp<walk::real_type>(pars["sd"]);
  return dust::pars_type<walk>(walk::shared_type{sd});
}

}

SEXP dust_walk_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp device_config) {
  return dust::r::dust_alloc<walk>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        device_config);
}

SEXP dust_walk_run(SEXP ptr, size_t step_end, bool device) {
  return dust::r::dust_run<walk>(ptr, step_end, device);
}

SEXP dust_walk_simulate(SEXP ptr, cpp11::sexp step_end, bool device) {
  return dust::r::dust_simulate<walk>(ptr, step_end, device);
}

SEXP dust_walk_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<walk>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_walk_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state) {
  return dust::r::dust_update_state<walk>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state);
}

SEXP dust_walk_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<walk>(ptr, r_index);
}

size_t dust_walk_step(SEXP ptr) {
  return dust::r::dust_step<walk>(ptr);
}

void dust_walk_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<walk>(ptr, r_index);
}

SEXP dust_walk_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<walk>(ptr, r_weights);
}

SEXP dust_walk_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<walk>(ptr, first_only, last_only);
}

SEXP dust_walk_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<walk>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_walk_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<walk>(ptr, data);
  return R_NilValue;
}

SEXP dust_walk_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<walk>(ptr, device);
}

SEXP dust_walk_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<walk>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_walk_capabilities() {
  return dust::r::dust_capabilities<walk>();
}

void dust_walk_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<walk>(ptr, n_threads);
}

int dust_walk_n_state(SEXP ptr) {
  return dust::r::dust_n_state<walk>(ptr);
}

cpp11::sexp dust_walk_device_info() {
  return dust::cuda::device_info<walk::real_type>();
}
