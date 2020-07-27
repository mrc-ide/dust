// Generated by dust (version 0.4.1) - do not edit
#include <dust/gpu/dust.hpp>
#include <dust/interface.hpp>

class sirs {
public:
  typedef float real_t;

  struct init_t {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t dt;
  };

  sirs(const init_t& data): internal(data) {
  }

  size_t size() {
    return 3;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(3);
    state[0] = internal.S0;
    state[1] = internal.I0;
    state[2] = internal.R0;
    return state;
  }

#ifdef __NVCC__
  __device__
#endif
  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t N = S + I + R;

    real_t p_SI = 1 - exp(- internal.beta * I / (real_t) N);
    real_t p_IR = 1 - exp(-(internal.gamma));
    real_t p_RS = 1 - exp(- internal.alpha);

    real_t n_SI = dust::distr::rbinom(rng_state, S, p_SI * internal.dt);
    real_t n_IR = dust::distr::rbinom(rng_state, I, p_IR * internal.dt);
    real_t n_RS = dust::distr::rbinom(rng_state, R, p_RS * internal.dt);

    state_next[0] = S - n_SI + n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR - n_RS;
  }

private:
  init_t internal;
};

#include <cpp11/list.hpp>
template <>
sirs::init_t dust_data<sirs>(cpp11::list data) {
  // Initial state values
  sirs::real_t I0 = 10.0;
  sirs::real_t S0 = 1000.0;
  sirs::real_t R0 = 0.0;

  // Default rates
  sirs::real_t alpha = 0.1;
  sirs::real_t beta = 0.2;
  sirs::real_t gamma = 0.1;

  // Time scaling
  sirs::real_t dt = 1.0;

  // Accept beta and gamma as optional elements
  SEXP r_beta = data["beta"];
  if (r_beta != R_NilValue) {
    beta = cpp11::as_cpp<sirs::real_t>(r_beta);
  }
  SEXP r_gamma = data["gamma"];
  if (r_gamma != R_NilValue) {
    gamma = cpp11::as_cpp<sirs::real_t>(r_gamma);
  }

  return sirs::init_t{S0, I0, R0, alpha, beta, gamma, dt};
}

SEXP dust_sirs_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                size_t n_threads, size_t seed) {
  return dust_alloc<sirs>(r_data, step, n_particles, n_threads, seed);
}

SEXP dust_sirs_run(SEXP ptr, size_t step_end) {
  return dust_run<sirs>(ptr, step_end);
}

SEXP dust_sirs_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<sirs>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_sirs_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<sirs>(ptr, r_state, r_step);
  return R_NilValue;
}

SEXP dust_sirs_reset(SEXP ptr, cpp11::list r_data, size_t step) {
  return dust_reset<sirs>(ptr, r_data, step);
}

SEXP dust_sirs_state(SEXP ptr, SEXP r_index) {
  return dust_state<sirs>(ptr, r_index);
}

size_t dust_sirs_step(SEXP ptr) {
  return dust_step<sirs>(ptr);
}

void dust_sirs_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<sirs>(ptr, r_index);
}

SEXP dust_sirs_rng_state(SEXP ptr) {
  return dust_rng_state<sirs>(ptr);
}
