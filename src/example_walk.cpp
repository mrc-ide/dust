#include <dust/dust.hpp>
#include <R.h>
#include <Rinternals.h>

#include "example_walk.h"

class walk {
public:
  walk(SEXP data) : sd(REAL(data)[0]) {
  }
  size_t size() const {
    return 1;
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

template <typename T>
T* read_r_pointer(SEXP r_ptr, bool closed_error) {
  void *ptr = NULL;
  if (TYPEOF(r_ptr) != EXTPTRSXP) {
    Rf_error("Expected an external pointer");
  }
  ptr = (void*) R_ExternalPtrAddr(r_ptr);
  if (!ptr && closed_error) {
    Rf_error("Pointer has been invalidated (perhaps serialised?)");
  }
  return static_cast<T*>(ptr);
}


extern "C" void test_walk_finalise(SEXP ptr) {
  dust_walk *obj = read_r_pointer<dust_walk>(ptr, false);
  if (obj) {
    delete obj;
  }
  if (ptr) {
    R_ClearExternalPtr(ptr);
  }
}


extern "C" SEXP test_walk_alloc(SEXP sd, SEXP r_n_particles, SEXP r_seed) {
  size_t n_particles = INTEGER(r_n_particles)[0];
  size_t seed = INTEGER(r_seed)[0];

  std::vector<size_t> index_y = {0};
  size_t n_threads = 1;


  dust_walk *d = new dust_walk(sd, index_y, n_threads, seed, n_particles);

  // Now, we need a simple way of getting this back into R as a
  // pointer

  SEXP r_ptr = PROTECT(R_MakeExternalPtr(d, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(r_ptr, test_walk_finalise);
  UNPROTECT(1);
  return r_ptr;
}


extern "C" SEXP test_walk_run(SEXP ptr, SEXP r_step_end) {
  const size_t step_end = INTEGER(r_step_end)[0];

  dust_walk *obj = read_r_pointer<dust_walk>(ptr, true);
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();

  std::vector<double> dat(n_state * n_particles);
  obj->state(dat);

  SEXP ret = PROTECT(allocMatrix(REALSXP, n_state, n_particles));
  memcpy(REAL(ret), &dat[0], dat.size() * sizeof(double));
  UNPROTECT(1);

  return ret;
}
