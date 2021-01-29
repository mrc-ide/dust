#include <vector>

class vectors {
public:
  typedef double real_t;
  typedef dust::no_data data_t;

  struct shared_t {
    int dim_r;
    int dim_x;
    std::vector<real_t> initial_x;
    real_t initial_y;
    int n;
  };

  struct internal_t {
    std::vector<real_t> r;
  };

  vectors(const dust::pars_t<vectors>& pars) :
    shared(pars.shared), internal(pars.internal) {
  }
  size_t size() {
    return 1 + shared->dim_x;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1 + shared->dim_x);
    state[0] = shared->initial_y;
    std::copy(shared->initial_x.begin(), shared->initial_x.end(),
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
  dust::shared_ptr<T> shared;
  internal_t internal;
};

template<>
dust::pars_t<vectors> dust_pars<vectors>(cpp11::list user) {
  typedef typename vectors::real_t real_t;

  auto shared = std::make_shared<vectors::shared_t>();
  shared->initial_y = 2;
  shared->n = 3;
  shared->dim_r = shared->n;
  shared->dim_x = shared->n;
  shared->initial_x = std::vector<real_t>(shared->dim_x);
  for (int i = 0; i < shared->dim_x; ++i) {
    shared->initial_x[i] = 1;
  }

  vectors::internal_t internal;
  internal.r = std::vector<real_t>(shared->dim_r);
  for (int i = 0; i < shared->dim_r; ++i) {
    internal.r[i] = i;
  }

  return dust::pars_t<vectors>(shared, internal);
}

template <>
cpp11::sexp dust_info<vectors>(const dust::pars_t<vectors>& pars) {
  cpp11::writable::list ret(2);
  ret[0] = cpp11::writable::integers({1});
  ret[1] = cpp11::writable::integers({pars.shared->dim_x});
  cpp11::writable::strings nms({"y", "x"});
  ret.names() = nms;
  return ret;
}
