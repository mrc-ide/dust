class walk {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    real_type sd;
    bool random_initial;
  };

  walk(const dust::pars_type<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    std::vector<real_type> ret = {0};
    if (shared->random_initial) {
      ret[0] = dust::random::normal<real_type>(rng_state, 0, shared->sd);
    }
    return ret;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
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
  const bool random_initial = pars["random_initial"] == R_NilValue ? false :
    cpp11::as_cpp<bool>(pars["random_initial"]);
  return dust::pars_type<walk>(walk::shared_type{sd, random_initial});
}

}
