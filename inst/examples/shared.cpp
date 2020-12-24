// Same as the arrays example but using shared memory. We split our
// internal storage into two chunks;
//
// * one is "internal" and can be arbitrarily modified; this is the
//   same as our normal internal state really and will be duplicated
//   for each particle.
// * the other is "shared" and will be stored as a "const reference";
//   we will guarantee that Dust holds a copy of the 'init_t' object
//   so that this reference stays valid.
#include <array>
#include <memory>

class arrays {
public:
  typedef double real_t;

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

  struct init_t {
    std::shared_ptr<const shared_t> shared;
    internal_t internal;
  };

  arrays(const init_t& data): shared(data.shared), internal(data.internal) {
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
  // We would *like* to make this a const shared_t& but that breaks
  // our assignment operator. We will need to make this instead a
  // little pointer class, which also means that we can remove the
  // special holding of data.
  std::shared_ptr<const shared_t> shared;
  internal_t internal;
};

template<>
arrays::init_t dust_data<arrays>(cpp11::list user) {
  arrays::shared_t shared;
  shared.initial_y = 2;
  shared.n = 3;
  shared.dim_r = shared.n;
  shared.dim_x = shared.n;
  for (int i = 0; i < shared.dim_x; ++i) {
    shared.initial_x[i] = 1;
  }

  arrays::internal_t internal;
  for (int i = 0; i < shared.dim_r; ++i) {
    internal.r[i] = i;
  }

  auto ptr = std::make_shared<const arrays::shared_t>(shared);

  return arrays::init_t{ptr, internal};
}

template <>
cpp11::sexp dust_info<arrays>(const arrays::init_t& data) {
  cpp11::writable::list ret(2);
  ret[0] = cpp11::writable::integers({1});
  ret[1] = cpp11::writable::integers({data.shared->dim_x});
  cpp11::writable::strings nms({"y", "x"});
  ret.names() = nms;
  return ret;
}
