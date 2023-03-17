class variable {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    size_t len;
    real_type mean;
    real_type sd;
  };

  variable(const dust::pars_type<variable>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return shared->len;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    std::vector<real_type> ret;
    for (size_t i = 0; i < shared->len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    for (size_t i = 0; i < shared->len; ++i) {
      state_next[i] = state[i] +
        dust::random::normal<real_type>(rng_state, shared->mean, shared->sd);
    }
  }

private:
  dust::shared_ptr<variable> shared;
};

namespace dust {
template <>
dust::pars_type<variable> dust_pars<variable>(cpp11::list pars) {
  using real_type = variable::real_type;
  const size_t len = cpp11::as_cpp<int>(pars["len"]);
  real_type mean = 0, sd = 1;

  SEXP r_mean = pars["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<real_type>(r_mean);
  }

  SEXP r_sd = pars["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<real_type>(r_sd);
  }

  variable::shared_type shared{len, mean, sd};
  return dust::pars_type<variable>(shared);
}

template <>
cpp11::sexp dust_info<variable>(const dust::pars_type<variable>& pars) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"len"_nm = pars.shared->len});
}

namespace gpu {

template <>
size_t shared_int_size<variable>(dust::shared_ptr<variable> shared) {
  return 1;
}

template <>
size_t shared_real_size<variable>(dust::shared_ptr<variable> shared) {
  return 2;
}

template <>
void shared_copy<variable>(dust::shared_ptr<variable> shared,
                           int * shared_int,
                           variable::real_type * shared_real) {
  using dust::gpu::shared_copy_data;
  using real_type = variable::real_type;
  shared_int = shared_copy_data<int>(shared_int, shared->len);
  shared_real = shared_copy_data<real_type>(shared_real, shared->mean);
  shared_real = shared_copy_data<real_type>(shared_real, shared->sd);
}

template <>
__device__
void update_gpu<variable>(size_t time,
                          const dust::gpu::interleaved<variable::real_type> state,
                          dust::gpu::interleaved<int> internal_int,
                          dust::gpu::interleaved<variable::real_type> internal_real,
                          const int * shared_int,
                          const variable::real_type * shared_real,
                          variable::rng_state_type& rng_state,
                          dust::gpu::interleaved<variable::real_type> state_next) {
  using real_type = variable::real_type;
  const size_t len = shared_int[0];
  const real_type mean = shared_real[0];
  const real_type sd = shared_real[1];
  for (size_t i = 0; i < len; ++i) {
    state_next[i] = state[i] +
      dust::random::normal<real_type>(rng_state, mean, sd);
  }
}

}
}
