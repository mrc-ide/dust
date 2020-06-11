#include <dust/dust.hpp>
#include <dust/util.hpp>

#include "example_walk.h"

class walk {
public:
  typedef SEXP init_t;
  walk(SEXP data) : sd(REAL(data)[0]) {
  }
  size_t size() const {
    return 1;
  }
  std::vector<double> initial(size_t step) {
    std::vector<double> ret = {0};
    return ret;
  }
  void update(size_t step, const std::vector<double> state, dust::RNG& rng,
              const size_t thread_idx, std::vector<double>& state_next) {
    double mean = state[0];
    state_next[0] = rng.rnorm(thread_idx, mean, sd);
  }
private:
  double sd;
};

typedef dust::Dust<walk> dust_walk;

extern "C" void test_walk_finalise(SEXP ptr) {
  dust_walk *obj = dust::util::read_r_pointer<dust_walk>(ptr, false);
  if (obj) {
    delete obj;
  }
  if (ptr) {
    R_ClearExternalPtr(ptr);
  }
}


extern "C" SEXP test_walk_alloc(SEXP sd, SEXP r_step,
                                SEXP r_n_particles, SEXP r_seed) {
  size_t n_particles = dust::util::as_size(r_n_particles, "n_particles");
  size_t seed = dust::util::as_size(r_seed, "seed");
  size_t step = dust::util::as_size(r_step, "step");

  std::vector<size_t> index_y = {0};
  size_t n_threads = 1;

  dust_walk *d = new dust_walk(sd, step, index_y, n_threads, seed, n_particles);

  SEXP r_ptr = PROTECT(R_MakeExternalPtr(d, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(r_ptr, test_walk_finalise);
  UNPROTECT(1);
  return r_ptr;
}


extern "C" SEXP test_walk_run(SEXP ptr, SEXP r_step_end) {
  const size_t step_end = dust::util::as_size(r_step_end, "step_end");

  dust_walk *obj = dust::util::read_r_pointer<dust_walk>(ptr, true);
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();

  std::vector<double> dat(n_state * n_particles);
  obj->state(dat);

  SEXP ret = PROTECT(allocMatrix(REALSXP, n_state, n_particles));
  memcpy(REAL(ret), dat.data(), dat.size() * sizeof(double));
  UNPROTECT(1);

  return ret;
}
