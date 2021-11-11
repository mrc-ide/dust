// Generated by dust (version 0.11.6) - do not edit
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_sir_capabilities();

[[cpp11::register]]
cpp11::sexp dust_sir_gpu_info();
[[cpp11::register]]
SEXP dust_cpu_sir_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config);

[[cpp11::register]]
SEXP dust_cpu_sir_run(SEXP ptr, size_t step_end);

[[cpp11::register]]
SEXP dust_cpu_sir_simulate(SEXP ptr, cpp11::sexp step_end);

[[cpp11::register]]
SEXP dust_cpu_sir_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_sir_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state);

[[cpp11::register]]
SEXP dust_cpu_sir_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_cpu_sir_step(SEXP ptr);

[[cpp11::register]]
void dust_cpu_sir_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_sir_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_cpu_sir_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_cpu_sir_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_cpu_sir_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_cpu_sir_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_cpu_sir_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot);

[[cpp11::register]]
void dust_cpu_sir_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_cpu_sir_n_state(SEXP ptr);
#include <dust/r/dust.hpp>

class sir {
public:
  typedef double real_type;
  struct data_type {
    real_type incidence;
  };
  typedef dust::no_internal internal_type;
  typedef dust::random::generator<real_type> rng_state_type;

  struct shared_type {
    real_type S0;
    real_type I0;
    real_type R0;
    real_type beta;
    real_type gamma;
    real_type dt;
    size_t freq;
    // Observation parameters
    real_type exp_noise;
  };

  sir(const dust::pars_type<sir>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 5;
  }

  std::vector<real_type> initial(size_t step) {
    std::vector<real_type> ret = {shared->S0, shared->I0, shared->R0, 0, 0};
    return ret;
  }

  void update(size_t step, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type S = state[0];
    real_type I = state[1];
    real_type R = state[2];
    real_type cumulative_incidence = state[3];

    real_type N = S + I + R;

    real_type p_SI = 1 - std::exp(-(shared->beta) * I / N);
    real_type p_IR = 1 - std::exp(-(shared->gamma));
    real_type n_IR = dust::random::binomial<real_type>(rng_state, I,
                                                       p_IR * shared->dt);
    real_type n_SI = dust::random::binomial<real_type>(rng_state, S,
                                                       p_SI * shared->dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cumulative_incidence + n_SI;
    // Little trick here to compute daily incidence by accumulating
    // incidence from the first day.
    state_next[4] = (step % shared->freq == 0) ? n_SI : state[4] + n_SI;
  }

  real_type compare_data(const real_type * state, const data_type& data,
                         rng_state_type& rng_state) {
    const real_type incidence_modelled = state[4];
    const real_type incidence_observed = data.incidence;
    const real_type lambda = incidence_modelled +
      dust::random::exponential(rng_state, shared->exp_noise);
    return dust::density::poisson(incidence_observed, lambda, true);
  }

private:
  dust::shared_ptr<sir> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {

template <>
dust::pars_type<sir> dust_pars<sir>(cpp11::list pars) {
  typedef sir::real_type real_type;
  // Initial state values
  // [[dust::param(I0, required = FALSE)]]
  real_type I0 = with_default(10, pars["I0"]);
  real_type S0 = 1000.0;
  real_type R0 = 0.0;

  // Rates, which can be set based on the provided pars
  // [[dust::param(beta, required = FALSE)]]
  real_type beta = with_default(0.2, pars["beta"]);
  // [[dust::param(gamma, required = FALSE)]]
  real_type gamma = with_default(0.1, pars["gamma"]);

  // Time scaling
  size_t freq = 4;
  real_type dt = 1.0 / static_cast<real_type>(freq);

  // Compare function
  // [[dust::param(exp_noise, required = FALSE)]]
  real_type exp_noise = with_default(1e6, pars["exp_noise"]);

  sir::shared_type shared{S0, I0, R0, beta, gamma, dt, freq, exp_noise};
  return dust::pars_type<sir>(shared);
}

template <>
cpp11::sexp dust_info<sir>(const dust::pars_type<sir>& pars) {
  using namespace cpp11::literals;
  // Information about state order
  cpp11::writable::strings vars({"S", "I", "R", "cases_cumul", "cases_inc"});
  // Information about parameter values
  cpp11::list p = cpp11::writable::list({"beta"_nm = pars.shared->beta,
                                         "gamma"_nm = pars.shared->gamma});
  return cpp11::writable::list({"vars"_nm = vars, "pars"_nm = p});
}

// The way that this is going to work is we will process a list
// *outside* of the C that will take (say) a df and convert it
// row-wise into a list with elements `step` and `data`, we will pass
// that in here. Then this function will be called once per data
// element to create the struct that will be used for future work.
template <>
sir::data_type dust_data<sir>(cpp11::list data) {
  return sir::data_type{cpp11::as_cpp<sir::real_type>(data["incidence"])};
}

}

cpp11::sexp dust_sir_capabilities() {
  return dust::r::dust_capabilities<sir>();
}

cpp11::sexp dust_sir_gpu_info() {
  return dust::gpu::r::gpu_info();
}
using model_cpu = dust::dust_cpu<sir>;

SEXP dust_cpu_sir_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                             cpp11::sexp r_n_particles, size_t n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config) {
  return dust::r::dust_cpu_alloc<sir>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config);
}

SEXP dust_cpu_sir_run(SEXP ptr, size_t step_end) {
  return dust::r::dust_run<model_cpu>(ptr, step_end);
}

SEXP dust_cpu_sir_simulate(SEXP ptr, cpp11::sexp step_end) {
  return dust::r::dust_simulate<model_cpu>(ptr, step_end);
}

SEXP dust_cpu_sir_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_cpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_cpu_sir_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state) {
  return dust::r::dust_update_state<model_cpu>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state);
}

SEXP dust_cpu_sir_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_cpu>(ptr, r_index);
}

size_t dust_cpu_sir_step(SEXP ptr) {
  return dust::r::dust_step<model_cpu>(ptr);
}

void dust_cpu_sir_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_sir_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_cpu>(ptr, r_weights);
}

SEXP dust_cpu_sir_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_cpu>(ptr, first_only, last_only);
}

SEXP dust_cpu_sir_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_cpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_cpu_sir_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<model_cpu>(ptr, data);
  return R_NilValue;
}

SEXP dust_cpu_sir_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_cpu>(ptr);
}

SEXP dust_cpu_sir_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot) {
  return dust::r::dust_filter<model_cpu>(ptr, save_trajectories, step_snapshot);
}

void dust_cpu_sir_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_cpu>(ptr, n_threads);
}

int dust_cpu_sir_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_cpu>(ptr);
}
