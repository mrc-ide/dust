#include <array>

class arrays {
public:
  typedef double real_t;
  typedef dust::no_data data_t;

  struct shared_t {
    int dim_r;
    int dim_x;
    int n;
    real_t initial_y;
    std::array<real_t, 3> initial_x;
  };

  struct internal_t {
    std::array<real_t, 3> r;
  };

  arrays(const dust::pars_t<arrays>& pars) :
    shared(pars.shared), internal(pars.internal) {
  }

  size_t size() {
    return 1 + shared->dim_x;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1 + shared->dim_x);
    state[0] = shared->initial_y;
    std::copy(shared->initial_x.begin(),
              shared->initial_x.end(),
              state.begin() + 1);
    return state;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t * x = state + 1;
    const real_t y = state[0];
    state_next[0] = y + shared->n;
    for (int i = 0; i < shared->dim_x; ++i) {
      state_next[i] = x[i] + internal.r[i];
    }
  }

private:
  std::shared_ptr<const shared_t> shared;
  internal_t internal;
};

template<>
dust::pars_t<arrays> dust_pars<arrays>(cpp11::list user) {
  auto shared = std::make_shared<arrays::shared_t>();

  shared->initial_y = 2;
  shared->n = 3;
  shared->dim_r = shared->n;
  shared->dim_x = shared->n;
  for (int i = 0; i < shared->dim_x; ++i) {
    shared->initial_x[i] = 1;
  }

  arrays::internal_t internal;
  for (int i = 0; i < shared->dim_r; ++i) {
    internal.r[i] = i;
  }

  return dust::pars_t<arrays>(shared, internal);
}

template <>
cpp11::sexp dust_info<arrays>(const dust::pars_t<arrays>& pars) {
  cpp11::writable::list ret(2);
  ret[0] = cpp11::writable::integers({1});
  ret[1] = cpp11::writable::integers({pars.shared->dim_x});
  cpp11::writable::strings nms({"y", "x"});
  ret.names() = nms;
  return ret;
}
