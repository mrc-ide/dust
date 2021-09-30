class walk {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t sd;
  };

  walk(const dust::pars_t<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret = {0};
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t& rng_state,
              real_t * state_next) {
    real_t mean = state[0];
    state_next[0] = dust::distr::rnorm(rng_state, mean, shared->sd);
  }

private:
  dust::shared_ptr<walk> shared;
};

namespace dust {

template <>
dust::pars_t<walk> dust_pars<walk>(cpp11::list pars) {
  walk::real_t sd = cpp11::as_cpp<walk::real_t>(pars["sd"]);
  return dust::pars_t<walk>(walk::shared_t{sd});
}

}
