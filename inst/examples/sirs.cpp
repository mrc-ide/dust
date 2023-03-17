class sirs {
public:
  using real_type = double;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  // __align__(16) is required before the data_type definition when using NVCC
  // This is so when loaded into shared memory it is aligned correctly
  struct __align__(16) data_type {
    double incidence;
  };

  struct shared_type {
    real_type S0;
    real_type I0;
    real_type R0;
    real_type alpha;
    real_type beta;
    real_type gamma;
    real_type dt;
    size_t freq;
    real_type exp_noise;
  };

  sirs(const dust::pars_type<sirs>& pars): shared(pars.shared) {
  }

  size_t size() const {
    return 4;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    std::vector<real_type> state(4);
    state[0] = shared->S0;
    state[1] = shared->I0;
    state[2] = shared->R0;
    state[3] = 0;
    return state;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type S = state[0];
    real_type I = state[1];
    real_type R = state[2];
    real_type N = S + I + R;

    real_type p_SI = 1 - exp(- shared->beta * I / N);
    real_type p_IR = 1 - exp(-(shared->gamma));
    real_type p_RS = 1 - exp(- shared->alpha);

    real_type dt = shared->dt;
    real_type n_SI = dust::random::binomial<real_type>(rng_state, S, p_SI * dt);
    real_type n_IR = dust::random::binomial<real_type>(rng_state, I, p_IR * dt);
    real_type n_RS = dust::random::binomial<real_type>(rng_state, R, p_RS * dt);

    state_next[0] = S - n_SI + n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR - n_RS;
    state_next[3] = (time % shared->freq == 0) ? n_SI : state[3] + n_SI;
  }

  real_type compare_data(const real_type * state, const data_type& data,
                         rng_state_type& rng_state) {
    const real_type incidence_modelled = state[3];
    const real_type incidence_observed = data.incidence;
    const real_type lambda = incidence_modelled +
      dust::random::exponential<real_type>(rng_state, shared->exp_noise);
    return dust::density::poisson(incidence_observed, lambda, true);
  }

private:
  dust::shared_ptr<sirs> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {
template <>
dust::pars_type<sirs> dust_pars<sirs>(cpp11::list pars) {
  // Initial state values
  sirs::real_type I0 = 10.0;
  sirs::real_type S0 = 1000.0;
  sirs::real_type R0 = 0.0;

  // Time scaling
  // [[dust::param(freq, required = FALSE, default = 1)]]
  size_t freq = std::max(1.0, with_default(1.0, pars["freq"]));
  sirs::real_type dt = 1 / static_cast<sirs::real_type>(freq);

  sirs::real_type exp_noise = 1e6;

  // [[dust::param(alpha, required = FALSE, default = 0.1)]]
  sirs::real_type alpha = with_default(0.1, pars["alpha"]);

  // [[dust::param(beta, required = FALSE, default = 0.2)]]
  sirs::real_type beta = with_default(0.2, pars["beta"]);

  // [[dust::param(gamma, required = FALSE, default = 0.1)]]
  sirs::real_type gamma = with_default(0.1, pars["gamma"]);

  sirs::shared_type shared{S0, I0, R0, alpha, beta, gamma, dt, freq, exp_noise};
  return dust::pars_type<sirs>(shared);
}

template <>
sirs::data_type dust_data<sirs>(cpp11::list data) {
  return sirs::data_type{cpp11::as_cpp<double>(data["incidence"])};
}

namespace gpu {

template <>
size_t shared_int_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 1;
}

template <>
size_t shared_real_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 5;
}

template <>
void shared_copy<sirs>(dust::shared_ptr<sirs> shared,
                       int * shared_int,
                       sirs::real_type * shared_real) {
  using real_type = sirs::real_type;
  using dust::gpu::shared_copy_data;
  shared_real = shared_copy_data<real_type>(shared_real, shared->alpha);
  shared_real = shared_copy_data<real_type>(shared_real, shared->beta);
  shared_real = shared_copy_data<real_type>(shared_real, shared->gamma);
  shared_real = shared_copy_data<real_type>(shared_real, shared->dt);
  shared_real = shared_copy_data<real_type>(shared_real, shared->exp_noise);
  shared_int = shared_copy_data<int>(shared_int, shared->freq);
}

template <>
__device__
void update_gpu<sirs>(size_t time,
                      const dust::gpu::interleaved<sirs::real_type> state,
                      dust::gpu::interleaved<int> internal_int,
                      dust::gpu::interleaved<sirs::real_type> internal_real,
                      const int * shared_int,
                      const sirs::real_type * shared_real,
                      sirs::rng_state_type& rng_state,
                      dust::gpu::interleaved<sirs::real_type> state_next) {
  using dust::random::binomial;
  using real_type = sirs::real_type;
  const real_type alpha = shared_real[0];
  const real_type beta = shared_real[1];
  const real_type gamma = shared_real[2];
  const real_type dt = shared_real[3];
  const int freq = shared_int[0];
  const real_type S = state[0];
  const real_type I = state[1];
  const real_type R = state[2];
  const real_type N = S + I + R;
  const real_type p_SI = 1 - exp(- beta * I / N);
  const real_type p_IR = 1 - exp(- gamma);
  const real_type p_RS = 1 - exp(- alpha);
  const real_type n_SI = binomial<real_type>(rng_state, S, p_SI * dt);
  const real_type n_IR = binomial<real_type>(rng_state, I, p_IR * dt);
  const real_type n_RS = binomial<real_type>(rng_state, R, p_RS * dt);
  state_next[0] = S - n_SI + n_RS;
  state_next[1] = I + n_SI - n_IR;
  state_next[2] = R + n_IR - n_RS;
  state_next[3] = (time % freq == 0) ? n_SI : state[3] + n_SI;
}

template <>
__device__
sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
                                  const sirs::data_type& data,
                                  dust::gpu::interleaved<int> internal_int,
                                  dust::gpu::interleaved<sirs::real_type> internal_real,
                                  const int * shared_int,
                                  const sirs::real_type * shared_real,
                                  sirs::rng_state_type& rng_state) {
  using real_type = sirs::real_type;
  const real_type exp_noise = shared_real[4];
  const real_type incidence_modelled = state[3];
  const real_type incidence_observed = data.incidence;
  const real_type lambda = incidence_modelled +
    dust::random::exponential<real_type>(rng_state, exp_noise);
  return dust::density::poisson(incidence_observed, lambda, true);
}

}
}
