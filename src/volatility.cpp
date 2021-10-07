// Generated by dust (version 0.9.22) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_volatility_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp device_config);

[[cpp11::register]]
SEXP dust_volatility_run(SEXP ptr, size_t step_end, bool device);

[[cpp11::register]]
SEXP dust_volatility_simulate(SEXP ptr, cpp11::sexp step_end, bool device);

[[cpp11::register]]
SEXP dust_volatility_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_volatility_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state);

[[cpp11::register]]
SEXP dust_volatility_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_volatility_step(SEXP ptr);

[[cpp11::register]]
void dust_volatility_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_volatility_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_volatility_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_volatility_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_volatility_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_volatility_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_volatility_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_volatility_capabilities();

[[cpp11::register]]
void dust_volatility_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_volatility_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_volatility_device_info();

#include <dust/dust.hpp>
#include <dust/interface/dust.hpp>

class volatility {
public:
  typedef double real_type;
  struct data_t {
    real_type observed;
  };
  typedef dust::no_internal internal_type;
  typedef dust::random::xoshiro256starstar_state rng_state_type;

  struct shared_type {
    real_type alpha;
    real_type sigma;
    real_type gamma;
    real_type tau;
    real_type x0;
  };

  volatility(const dust::pars_type<volatility>& pars) : shared(pars.shared) {
  }

  size_t size() {
    return 1;
  }

  std::vector<real_type> initial(size_t step) {
    std::vector<real_type> state(1);
    state[0] = shared->x0;
    return state;
  }

  void update(size_t step, const real_type * state,
              rng_state_type& rng_state, real_type * state_next) {
    const real_type x = state[0];
    state_next[0] = shared->alpha * x +
      shared->sigma * dust::random::normal<real_type>(rng_state, 0, 1);
  }

  real_type compare_data(const real_type * state, const data_t& data,
                         rng_state_type& rng_state) {
    return dust::density::normal(data.observed, shared->gamma * state[0],
                                 shared->tau, true);
  }

private:
  dust::shared_ptr<volatility> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {

template <>
dust::pars_type<volatility> dust_pars<volatility>(cpp11::list pars) {
  typedef volatility::real_type real_type;
  real_type x0 = 0;
  real_type alpha = with_default(0.91, pars["alpha"]);
  real_type sigma = with_default(1, pars["sigma"]);
  real_type gamma = with_default(1, pars["gamma"]);
  real_type tau = with_default(1, pars["tau"]);

  volatility::shared_type shared{alpha, sigma, gamma, tau, x0};
  return dust::pars_type<volatility>(shared);
}

template <>
volatility::data_t dust_data<volatility>(cpp11::list data) {
  return volatility::data_t{cpp11::as_cpp<double>(data["observed"])};
}

}

SEXP dust_volatility_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp device_config) {
  return dust::r::dust_alloc<volatility>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        device_config);
}

SEXP dust_volatility_run(SEXP ptr, size_t step_end, bool device) {
  return dust::r::dust_run<volatility>(ptr, step_end, device);
}

SEXP dust_volatility_simulate(SEXP ptr, cpp11::sexp step_end, bool device) {
  return dust::r::dust_simulate<volatility>(ptr, step_end, device);
}

SEXP dust_volatility_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<volatility>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_volatility_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state) {
  return dust::r::dust_update_state<volatility>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state);
}

SEXP dust_volatility_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<volatility>(ptr, r_index);
}

size_t dust_volatility_step(SEXP ptr) {
  return dust::r::dust_step<volatility>(ptr);
}

void dust_volatility_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<volatility>(ptr, r_index);
}

SEXP dust_volatility_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<volatility>(ptr, r_weights);
}

SEXP dust_volatility_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<volatility>(ptr, first_only, last_only);
}

SEXP dust_volatility_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<volatility>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_volatility_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<volatility>(ptr, data);
  return R_NilValue;
}

SEXP dust_volatility_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<volatility>(ptr, device);
}

SEXP dust_volatility_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<volatility>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_volatility_capabilities() {
  return dust::r::dust_capabilities<volatility>();
}

void dust_volatility_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<volatility>(ptr, n_threads);
}

int dust_volatility_n_state(SEXP ptr) {
  return dust::r::dust_n_state<volatility>(ptr);
}

cpp11::sexp dust_volatility_device_info() {
  return dust::cuda::device_info<volatility::real_type>();
}
