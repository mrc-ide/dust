// Generated by dust (version 0.4.8) - do not edit
#include <iostream>
#include <dust/dust.hpp>
#include <dust/interface.hpp>

class sir {
public:
  typedef double real_t;
  struct init_t {
    double S0;
    double I0;
    double R0;
    double beta;
    double gamma;
    double dt;
  };

  sir(const init_t& data) : data_(data) {
  }

  size_t size() const {
    return 4;
  }

  std::vector<double> initial(size_t step) {
    std::vector<double> ret = {data_.S0, data_.I0, data_.R0, 0};
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    double S = state[0];
    double I = state[1];
    double R = state[2];
    double cumulative_incidence = state[3];

    double N = S + I + R;

    double p_SI = 1 - std::exp(-(data_.beta) * I / N);
    double p_IR = 1 - std::exp(-(data_.gamma));
    double n_IR = dust::distr::rbinom(rng_state, I, p_IR * data_.dt);
    double n_SI = dust::distr::rbinom(rng_state, S, p_SI * data_.dt);

    state_next[0] = S - n_SI;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR;
    state_next[3] = cumulative_incidence + n_SI;
  }

private:
  init_t data_;
};

#include <cpp11/list.hpp>

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

template <>
sir::init_t dust_data<sir>(cpp11::list data) {
  // Initial state values
  double I0 = 10.0;
  double S0 = 1000.0;
  double R0 = 0.0;

  // Rates, which can be set based on the provided data
  double beta = with_default(0.2, data["beta"]);
  double gamma = with_default(0.1, data["gamma"]);

  // Time scaling
  double dt = 0.25;

  return sir::init_t{S0, I0, R0, beta, gamma, dt};
}

template <>
cpp11::sexp dust_info<sir>(const sir::init_t& data) {
  using namespace cpp11::literals;
  // Information about state order
  cpp11::writable::strings vars({"S", "I", "R", "inc"});
  // Information about parameter values
  cpp11::list pars = cpp11::writable::list({"beta"_nm = data.beta,
                                            "gamma"_nm = data.gamma});
  return cpp11::writable::list({"vars"_nm = vars, "pars"_nm = pars});
}

[[cpp11::register]]
SEXP dust_sir_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                size_t n_threads, cpp11::sexp r_seed) {
  return dust_alloc<sir>(r_data, step, n_particles, n_threads, r_seed);
}

[[cpp11::register]]
SEXP dust_sir_run(SEXP ptr, size_t step_end) {
  return dust_run<sir>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_sir_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<sir>(ptr, r_index);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<sir>(ptr, r_state, r_step);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sir_reset(SEXP ptr, cpp11::list r_data, size_t step) {
  return dust_reset<sir>(ptr, r_data, step);
}

[[cpp11::register]]
SEXP dust_sir_state(SEXP ptr, SEXP r_index) {
  return dust_state<sir>(ptr, r_index);
}

[[cpp11::register]]
size_t dust_sir_step(SEXP ptr) {
  return dust_step<sir>(ptr);
}

[[cpp11::register]]
void dust_sir_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<sir>(ptr, r_index);
}

[[cpp11::register]]
SEXP dust_sir_rng_state(SEXP ptr, bool first_only) {
  return dust_rng_state<sir>(ptr, first_only);
}

[[cpp11::register]]
SEXP dust_sir_simulate(cpp11::sexp r_steps,
                            cpp11::list r_data,
                            cpp11::doubles_matrix r_state,
                            cpp11::sexp r_index,
                            const size_t n_threads,
                            cpp11::sexp r_seed) {
  return dust_simulate<sir>(r_steps, r_data, r_state, r_index,
                                 n_threads, r_seed);
}

[[cpp11::register]]
bool dust_sir_has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}
