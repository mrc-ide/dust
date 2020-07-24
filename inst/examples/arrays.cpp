class arrays {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    int_t dim_r;
    int_t dim_x;
    std::vector<real_t> initial_x;
    real_t initial_y;
    int_t n;
    std::vector<real_t> r;
  };
  arrays(const init_t& data): internal(data) {
  }
  size_t size() {
    return 1 + internal.dim_x;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1 + internal.dim_x);
    state[0] = internal.initial_y;
    std::copy(internal.initial_x.begin(), internal.initial_x.end(), state.begin() + 1);
    return state;
  }
  #ifdef __NVCC__
  __device__
  #endif
  void update(size_t step, const real_t * state, dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t * x = state + 1;
    const real_t y = state[0];
    state_next[0] = y + internal.n;
    for (int_t i = 1; i <= internal.dim_x; ++i) {
      state_next[1 + i - 1] = x[i - 1] + internal.r[i - 1];
    }
  }

private:
  init_t internal;
};

template<>
arrays::init_t dust_data<arrays>(cpp11::list user) {
  typedef typename arrays::real_t real_t;
  typedef typename arrays::int_t int_t;
  arrays::init_t internal;
  internal.initial_y = 2;
  internal.n = 3;
  internal.dim_r = internal.n;
  internal.dim_x = internal.n;
  internal.initial_x = std::vector<real_t>(internal.dim_x);
  internal.r = std::vector<real_t>(internal.dim_r);
  for (int_t i = 1; i <= internal.dim_x; ++i) {
    internal.initial_x[i - 1] = 1;
  }
  for (int_t i = 1; i <= internal.dim_r; ++i) {
    internal.r[i - 1] = i;
  }
  return internal;
}

template <>
cpp11::sexp dust_info<arrays>(const arrays::init_t& internal) {
  cpp11::writable::list ret(2);
  ret[0] = cpp11::writable::integers({1});
  ret[1] = cpp11::writable::integers({internal.dim_x});
  cpp11::writable::strings nms({"y", "x"});
  ret.names() = nms;
  return ret;
}
