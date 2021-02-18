class variable {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    size_t len;
    real_t mean;
    real_t sd;
  };

  variable(const dust::pars_t<variable>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return shared->len;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret;
    for (size_t i = 0; i < shared->len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    for (size_t i = 0; i < shared->len; ++i) {
      state_next[i] =
        dust::distr::rnorm(rng_state, state[i] + shared->mean, shared->sd);
    }
  }

private:
  dust::shared_ptr<variable> shared;
};

// This is required to activate gpu support. We can possibly do this
// automatically based on the existance of the template below though.
namespace dust {
template <>
struct has_gpu_support<variable> : std::true_type {};

template <>
size_t device_shared_size_int<variable>(dust::shared_ptr<variable> shared) {
  return 1;
}

template <>
size_t device_shared_size_real<variable>(dust::shared_ptr<variable> shared) {
  return 2;
}

template <>
void device_shared_copy<variable>(dust::shared_ptr<variable> shared,
                                  int * shared_int,
                                  variable::real_t * shared_real) {
  shared_int[0] = shared->len;
  shared_real[0] = shared->mean;
  shared_real[1] = shared->sd;
}

}

template <>
void update_device<variable>(size_t step,
                             const dust::interleaved<variable::real_t> state,
                             dust::interleaved<int> internal_int,
                             dust::interleaved<variable::real_t> internal_real,
                             const int * shared_int,
                             const variable::real_t * shared_real,
                             dust::rng_state_t<variable::real_t>& rng_state,
                             dust::interleaved<variable::real_t> state_next) {
  typedef variable::real_t real_t;
  const size_t len = shared_int[0];
  const real_t mean = shared_real[0];
  const real_t sd = shared_real[1];
  for (size_t i = 0; i < len; ++i) {
    state_next[i] = dust::distr::rnorm(rng_state, state[i] + mean, sd);
  }
}

#include <cpp11/list.hpp>
template <>
dust::pars_t<variable> dust_pars<variable>(cpp11::list pars) {
  typedef variable::real_t real_t;
  const size_t len = cpp11::as_cpp<int>(pars["len"]);
  real_t mean = 0, sd = 1;

  SEXP r_mean = pars["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<real_t>(r_mean);
  }

  SEXP r_sd = pars["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<real_t>(r_sd);
  }

  variable::shared_t shared{len, mean, sd};
  return dust::pars_t<variable>(shared);
}
