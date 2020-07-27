#include <dust/gpu/dust.hpp>
#include <dust/interface.hpp>

class volatility {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    real_t alpha;
    real_t sigma;
    real_t x0;
  };
  volatility(const init_t& data): data_(data) {
  }
  size_t size() {
    return 1;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1);
    state[0] = data_.x0;
    return state;
  }
  #ifdef __NVCC__
  __device__
  #endif
  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t x = state[0];
    state_next[0] = data_.alpha * x +
      data_.sigma * dust::distr::rnorm(rng_state, 0, 1);
  }

private:
  init_t data_;
};

#include <cpp11/list.hpp>
template <>
volatility::init_t dust_data<volatility>(cpp11::list data) {
  volatility::real_t x0 = 0;
  volatility::real_t alpha = 0.91;
  volatility::real_t sigma = 1;

  SEXP r_alpha = data["alpha"];
  if (r_alpha != R_NilValue) {
    alpha = cpp11::as_cpp<volatility::real_t>(r_alpha);
  }
  SEXP r_sigma = data["sigma"];
  if (r_sigma != R_NilValue) {
    sigma = cpp11::as_cpp<volatility::real_t>(r_sigma);
  }

  return volatility::init_t{alpha, sigma, x0};
}

SEXP dust_volatilitygpu_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                size_t n_threads, size_t seed) {
  return dust_alloc<volatility>(r_data, step, n_particles, n_threads, seed);
}

SEXP dust_volatilitygpu_run(SEXP ptr, size_t step_end) {
  return dust_run<volatility>(ptr, step_end);
}

SEXP dust_volatilitygpu_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<volatility>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_volatilitygpu_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<volatility>(ptr, r_state, r_step);
  return R_NilValue;
}

SEXP dust_volatilitygpu_reset(SEXP ptr, cpp11::list r_data, size_t step) {
  return dust_reset<volatility>(ptr, r_data, step);
}

SEXP dust_volatilitygpu_state(SEXP ptr, SEXP r_index) {
  return dust_state<volatility>(ptr, r_index);
}

size_t dust_volatilitygpu_step(SEXP ptr) {
  return dust_step<volatility>(ptr);
}

void dust_volatilitygpu_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<volatility>(ptr, r_index);
}

SEXP dust_volatilitygpu_rng_state(SEXP ptr) {
  return dust_rng_state<volatility>(ptr);
}
