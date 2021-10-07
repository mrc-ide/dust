class variable {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_type;
  typedef dust::random::xoshiro256starstar_state rng_state_type;

  struct shared_type {
    size_t len;
    real_t mean;
    real_t sd;
  };

  variable(const dust::pars_type<variable>& pars) : shared(pars.shared) {
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

  void update(size_t step, const real_t * state, rng_state_type& rng_state,
              real_t * state_next) {
    for (size_t i = 0; i < shared->len; ++i) {
      state_next[i] = state[i] +
        dust::random::normal<real_t>(rng_state, shared->mean, shared->sd);
    }
  }

private:
  dust::shared_ptr<variable> shared;
};

namespace dust {
template <>
dust::pars_type<variable> dust_pars<variable>(cpp11::list pars) {
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

  variable::shared_type shared{len, mean, sd};
  return dust::pars_type<variable>(shared);
}

template <>
struct has_gpu_support<variable> : std::true_type {};

namespace cuda {

template <>
size_t device_shared_int_size<variable>(dust::shared_ptr<variable> shared) {
  return 1;
}

template <>
size_t device_shared_real_size<variable>(dust::shared_ptr<variable> shared) {
  return 2;
}

template <>
void device_shared_copy<variable>(dust::shared_ptr<variable> shared,
                                  int * shared_int,
                                  variable::real_t * shared_real) {
  using dust::cuda::shared_copy;
  typedef variable::real_t real_t;
  shared_int = shared_copy<int>(shared_int, shared->len);
  shared_real = shared_copy<real_t>(shared_real, shared->mean);
  shared_real = shared_copy<real_t>(shared_real, shared->sd);
}

template <>
DEVICE
void update_device<variable>(size_t step,
                             const dust::cuda::interleaved<variable::real_t> state,
                             dust::cuda::interleaved<int> internal_int,
                             dust::cuda::interleaved<variable::real_t> internal_real,
                             const int * shared_int,
                             const variable::real_t * shared_real,
                             variable::rng_state_type& rng_state,
                             dust::cuda::interleaved<variable::real_t> state_next) {
  typedef variable::real_t real_t;
  const size_t len = shared_int[0];
  const real_t mean = shared_real[0];
  const real_t sd = shared_real[1];
  for (size_t i = 0; i < len; ++i) {
    state_next[i] = state[i] +
      dust::random::normal<real_t>(rng_state, mean, sd);
  }
}

}
}
