// Generated by dust (version 0.13.3) - do not edit
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_volatility_capabilities();

[[cpp11::register]]
cpp11::sexp dust_volatility_gpu_info();
[[cpp11::register]]
SEXP dust_cpu_volatility_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                         cpp11::sexp r_n_particles, int n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config, cpp11::sexp ode_control);

[[cpp11::register]]
SEXP dust_cpu_volatility_run(SEXP ptr, cpp11::sexp r_time_end);

[[cpp11::register]]
SEXP dust_cpu_volatility_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_cpu_volatility_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_volatility_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_cpu_volatility_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
SEXP dust_cpu_volatility_time(SEXP ptr);

[[cpp11::register]]
void dust_cpu_volatility_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_volatility_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_cpu_volatility_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_cpu_volatility_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_cpu_volatility_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_cpu_volatility_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_cpu_volatility_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_cpu_volatility_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_cpu_volatility_n_state(SEXP ptr);

[[cpp11::register]]
void dust_cpu_volatility_set_stochastic_schedule(SEXP ptr, SEXP time);

[[cpp11::register]]
SEXP dust_cpu_volatility_ode_statistics(SEXP ptr);
#include <dust/r/dust.hpp>

class volatility {
public:
  using real_type = double;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct data_type {
    real_type observed;
  };

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

  std::vector<real_type> initial(size_t time) {
    std::vector<real_type> state(1);
    state[0] = shared->x0;
    return state;
  }

  void update(size_t time, const real_type * state,
              rng_state_type& rng_state, real_type * state_next) {
    const real_type x = state[0];
    state_next[0] = shared->alpha * x +
      shared->sigma * dust::random::normal<real_type>(rng_state, 0, 1);
  }

  real_type compare_data(const real_type * state, const data_type& data,
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
  using real_type = volatility::real_type;
  real_type x0 = 0;
  real_type alpha = with_default(0.91, pars["alpha"]);
  real_type sigma = with_default(1, pars["sigma"]);
  real_type gamma = with_default(1, pars["gamma"]);
  real_type tau = with_default(1, pars["tau"]);

  volatility::shared_type shared{alpha, sigma, gamma, tau, x0};
  return dust::pars_type<volatility>(shared);
}

template <>
volatility::data_type dust_data<volatility>(cpp11::list data) {
  return volatility::data_type{cpp11::as_cpp<double>(data["observed"])};
}

}

cpp11::sexp dust_volatility_capabilities() {
  return dust::r::dust_capabilities<volatility>();
}

cpp11::sexp dust_volatility_gpu_info() {
  return dust::gpu::r::gpu_info();
}
using model_cpu = dust::dust_cpu<volatility>;

SEXP dust_cpu_volatility_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                             cpp11::sexp r_n_particles, int n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
  return dust::r::dust_cpu_alloc<volatility>(r_pars, pars_multi, r_time, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config, ode_control);
}

SEXP dust_cpu_volatility_run(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_run<model_cpu>(ptr, r_time_end);
}

SEXP dust_cpu_volatility_simulate(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_simulate<model_cpu>(ptr, r_time_end);
}

SEXP dust_cpu_volatility_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_cpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_cpu_volatility_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
  return dust::r::dust_update_state<model_cpu>(ptr, r_pars, r_state, r_time,
                                                      r_set_initial_state, index, reset_step_size);
}

SEXP dust_cpu_volatility_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_volatility_time(SEXP ptr) {
  return dust::r::dust_time<model_cpu>(ptr);
}

void dust_cpu_volatility_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_volatility_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_cpu>(ptr, r_weights);
}

SEXP dust_cpu_volatility_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_cpu>(ptr, first_only, last_only);
}

SEXP dust_cpu_volatility_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_cpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_cpu_volatility_set_data(SEXP ptr, cpp11::list data,
                                       bool shared) {
  dust::r::dust_set_data<model_cpu>(ptr, data, shared);
  return R_NilValue;
}

SEXP dust_cpu_volatility_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_cpu>(ptr);
}

SEXP dust_cpu_volatility_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood) {
  return dust::r::dust_filter<model_cpu>(ptr, time_end,
                                                save_trajectories,
                                                time_snapshot,
                                                min_log_likelihood);
}

void dust_cpu_volatility_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_cpu>(ptr, n_threads);
}

int dust_cpu_volatility_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_cpu>(ptr);
}

void dust_cpu_volatility_set_stochastic_schedule(SEXP ptr, SEXP time) {
  dust::r::dust_set_stochastic_schedule<model_cpu>(ptr, time);
}

SEXP dust_cpu_volatility_ode_statistics(SEXP ptr) {
  return dust::r::dust_ode_statistics<model_cpu>(ptr);
}
