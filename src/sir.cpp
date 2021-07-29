// Generated by dust (version 0.9.11) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_sir_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_config);

[[cpp11::register]]
SEXP dust_sir_run(SEXP ptr, size_t step_end, bool device,
                       bool deterministic);

[[cpp11::register]]
SEXP dust_sir_simulate(SEXP ptr, cpp11::sexp step_end, bool device,
                            bool deterministic);

[[cpp11::register]]
SEXP dust_sir_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step,
                             bool deterministic);

[[cpp11::register]]
SEXP dust_sir_reset(SEXP ptr, cpp11::list r_pars, size_t step);

[[cpp11::register]]
SEXP dust_sir_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_sir_step(SEXP ptr);

[[cpp11::register]]
void dust_sir_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_sir_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_sir_set_pars(SEXP ptr, cpp11::list r_pars);

[[cpp11::register]]
SEXP dust_sir_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_sir_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_sir_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_sir_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_sir_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_sir_capabilities();

[[cpp11::register]]
void dust_sir_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_sir_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_sir_device_info();

#include <dust/dust.hpp>
#include <dust/interface.hpp>

class sir {
public:
  typedef double real_t;
  struct data_t {
    real_t incidence;
  };
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t beta;
    real_t gamma;
    real_t dt;
    size_t freq;
    // Observation parameters
    real_t exp_noise;
  };

  sir(const dust::pars_t<sir>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 5;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {shared->S0, shared->I0, shared->R0, 0, 0};
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t cumulative_incidence = state[3];

    real_t N = S + I + R;

    real_t p_SI = 1 - std::exp(-(shared->beta) * I / N);
    real_t p_IR = 1 - std::exp(-(shared->gamma));
    real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * shared->dt);
    real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * shared->dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cumulative_incidence + n_SI;
    // Little trick here to compute daily incidence by accumulating
    // incidence from the first day.
    state_next[4] = (step % shared->freq == 0) ? n_SI : state[4] + n_SI;
  }

  real_t compare_data(const real_t * state, const data_t& data,
                      dust::rng_state_t<real_t>& rng_state) {
    const real_t incidence_modelled = state[4];
    const real_t incidence_observed = data.incidence;
    const real_t lambda = incidence_modelled +
      dust::distr::rexp(rng_state, shared->exp_noise);
    return dust::dpois(incidence_observed, lambda, true);
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
dust::pars_t<sir> dust_pars<sir>(cpp11::list pars) {
  typedef sir::real_t real_t;
  // Initial state values
  // [[dust::param(I0, required = FALSE)]]
  real_t I0 = with_default(10, pars["I0"]);
  real_t S0 = 1000.0;
  real_t R0 = 0.0;

  // Rates, which can be set based on the provided pars
  // [[dust::param(beta, required = FALSE)]]
  real_t beta = with_default(0.2, pars["beta"]);
  // [[dust::param(gamma, required = FALSE)]]
  real_t gamma = with_default(0.1, pars["gamma"]);

  // Time scaling
  size_t freq = 4;
  real_t dt = 1.0 / static_cast<real_t>(freq);

  // Compare function
  // [[dust::param(exp_noise, required = FALSE)]]
  real_t exp_noise = with_default(1e6, pars["exp_noise"]);

  sir::shared_t shared{S0, I0, R0, beta, gamma, dt, freq, exp_noise};
  return dust::pars_t<sir>(shared);
}

template <>
cpp11::sexp dust_info<sir>(const dust::pars_t<sir>& pars) {
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
sir::data_t dust_data<sir>(cpp11::list data) {
  return sir::data_t{cpp11::as_cpp<sir::real_t>(data["incidence"])};
}

}

SEXP dust_sir_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_config) {
  return dust::r::dust_alloc<sir>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, device_config);
}

SEXP dust_sir_run(SEXP ptr, size_t step_end, bool device,
                       bool deterministic) {
  return dust::r::dust_run<sir>(ptr, step_end, device, deterministic);
}

SEXP dust_sir_simulate(SEXP ptr, cpp11::sexp step_end, bool device,
                            bool deterministic) {
  return dust::r::dust_simulate<sir>(ptr, step_end, device,
                                           deterministic);
}

SEXP dust_sir_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<sir>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step,
                             bool deterministic) {
  dust::r::dust_set_state<sir>(ptr, r_state, r_step, deterministic);
  return R_NilValue;
}

SEXP dust_sir_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust::r::dust_reset<sir>(ptr, r_pars, step);
}

SEXP dust_sir_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<sir>(ptr, r_index);
}

size_t dust_sir_step(SEXP ptr) {
  return dust::r::dust_step<sir>(ptr);
}

void dust_sir_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<sir>(ptr, r_index);
}

SEXP dust_sir_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<sir>(ptr, r_weights);
}

SEXP dust_sir_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust::r::dust_set_pars<sir>(ptr, r_pars);
}

SEXP dust_sir_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<sir>(ptr, first_only, last_only);
}

SEXP dust_sir_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<sir>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_sir_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<sir>(ptr, data);
  return R_NilValue;
}

SEXP dust_sir_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<sir>(ptr, device);
}

SEXP dust_sir_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<sir>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_sir_capabilities() {
  return dust::r::dust_capabilities<sir>();
}

void dust_sir_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<sir>(ptr, n_threads);
}

int dust_sir_n_state(SEXP ptr) {
  return dust::r::dust_n_state<sir>(ptr);
}

cpp11::sexp dust_sir_device_info() {
  return dust::cuda::device_info<sir::real_t>();
}
