class walk {
public:
  typedef double real_type;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_type;
  typedef dust::random::xoshiro256starstar_state rng_state_type;

  struct shared_type {
    real_type sd;
  };

  walk(const dust::pars_type<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_type> initial(size_t step) {
    std::vector<real_type> ret = {0};
    return ret;
  }

  void update(size_t step, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type mean = state[0];
    state_next[0] = dust::random::normal<real_type>(rng_state, mean, shared->sd);
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
