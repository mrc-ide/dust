// Generated by dust (version 0.9.10) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_walk_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_id);

[[cpp11::register]]
SEXP dust_walk_run(SEXP ptr, size_t step_end, bool device);

[[cpp11::register]]
SEXP dust_walk_simulate(SEXP ptr, cpp11::sexp step_end, bool device);

[[cpp11::register]]
SEXP dust_walk_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_walk_set_state(SEXP ptr, SEXP r_state, SEXP r_step);

[[cpp11::register]]
SEXP dust_walk_reset(SEXP ptr, cpp11::list r_pars, size_t step);

[[cpp11::register]]
SEXP dust_walk_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_walk_step(SEXP ptr);

[[cpp11::register]]
void dust_walk_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_walk_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_walk_set_pars(SEXP ptr, cpp11::list r_pars);

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
#include <dust/interface.hpp>

class walk {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t sd;
  };

  walk(const dust::pars_t<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {0};
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    real_t mean = state[0];
    state_next[0] = dust::distr::rnorm(rng_state, mean, shared->sd);
  }

private:
  dust::shared_ptr<walk> shared;
};

namespace dust {

template <>
dust::pars_t<walk> dust_pars<walk>(cpp11::list pars) {
  walk::real_t sd = cpp11::as_cpp<walk::real_t>(pars["sd"]);
  return dust::pars_t<walk>(walk::shared_t{sd});
}

}

SEXP dust_walk_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_id) {
  return dust::r::dust_alloc<walk>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, device_id);
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

SEXP dust_walk_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust::r::dust_set_state<walk>(ptr, r_state, r_step);
  return R_NilValue;
}

SEXP dust_walk_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust::r::dust_reset<walk>(ptr, r_pars, step);
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

SEXP dust_walk_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust::r::dust_set_pars<walk>(ptr, r_pars);
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
  return dust::cuda::device_info<walk::real_t>();
}
