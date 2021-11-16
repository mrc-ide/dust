// A version of inst/examples/walk.cpp but with stochastic
// time-varying initialisation, which we'll use to explore some
// pathalogical initialisation cases.
class walk {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    real_type sd;
  };

  walk(const dust::pars_type<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_type> initial(size_t step) {
    return std::vector<real_type>{static_cast<real_type>(step)};
  }

  void update(size_t step, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    state_next[0] = state[0] +
      dust::random::normal<real_type>(rng_state, 0, shared->sd);
  }

private:
  dust::shared_ptr<walk> shared;
};

namespace dust {

template <>
dust::pars_type<walk> dust_pars<walk>(cpp11::list pars) {
  walk::real_type sd = cpp11::as_cpp<walk::real_type>(pars["sd"]);
  return dust::pars_type<walk>(walk::shared_type{sd});
}

}
