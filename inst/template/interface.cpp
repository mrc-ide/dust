#include <dust/dust.hpp>
#include <dust/util.hpp>

{{model}}

extern "C" void {{name}}_finalise(SEXP ptr) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, false);
  if (obj) {
    delete obj;
  }
  if (ptr) {
    R_ClearExternalPtr(ptr);
  }
}

extern "C" SEXP {{name}}_alloc(SEXP data, SEXP r_n_particles, SEXP r_seed) {
  size_t n_particles = dust::util::as_size(r_n_particles, "n_particles");
  size_t seed = dust::util::as_size(r_seed, "seed");

  std::vector<size_t> index_y = {0};
  size_t n_threads = 1;

  dust::Dust<{{type}}> *d =
    new dust::Dust<{{type}}>(data, index_y, n_threads, seed, n_particles);

  SEXP r_ptr = PROTECT(R_MakeExternalPtr(d, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(r_ptr, {{name}}_finalise);
  UNPROTECT(1);
  return r_ptr;
}

extern "C" SEXP {{name}}_run(SEXP ptr, SEXP r_step_end) {
  const size_t step_end = dust::util::as_size(r_step_end, "step_end");

  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);
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

extern "C" SEXP {{name}}_reset(SEXP ptr, SEXP data) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);
  obj->reset(data);
  return R_NilValue;
}
