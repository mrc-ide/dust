#include "interface.h"

#include <stdbool.h>

// R interface
SEXP r_particle_alloc(SEXP r_create, SEXP r_update, SEXP r_free,
                      SEXP r_y, SEXP user, SEXP r_index_y) {
  model_create * f_create = (model_create*) ptr_fn_get(r_create);
  model_update * f_update = (model_update*) ptr_fn_get(r_update);
  model_free * f_free = (model_free*) ptr_fn_get(r_free);

  size_t n_index_y = length(r_index_y);
  size_t *index_y = (size_t*) R_alloc(n_index_y, sizeof(size_t));
  for (size_t i = 0; i < n_index_y; ++i) {
    index_y[i] = INTEGER(r_index_y)[i] - 1;
  }
  size_t n_y = length(r_y);
  particle * obj = particle_alloc(f_create, f_update, f_free,
                                  n_y, REAL(r_y), user,
                                  n_index_y, index_y);
  SEXP r_ptr = PROTECT(R_MakeExternalPtr(obj, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(r_ptr, particle_finalise);
  UNPROTECT(1);
  return r_ptr;
}

void particle_finalise(SEXP r_ptr) {
  particle *obj = (particle*) read_r_pointer(r_ptr, 0);
  if (r_ptr) {
    particle_free(obj);
    R_ClearExternalPtr(r_ptr);
  }
}

SEXP r_particle_run(SEXP r_ptr, SEXP r_step_end, SEXP r_seed) {
  particle *obj = (particle*) read_r_pointer(r_ptr, true);
  size_t step_end = (size_t) INTEGER(r_step_end)[0];
  uint64_t seed = (uint64_t) INTEGER(r_seed)[0];
  
  // Note: running a single particle allocs a rng each
  // time that it is called
  RNG* rng = C_RNG_alloc(1, seed);

  // Actual running bit
  particle_run(obj, step_end, rng, 0);

  C_RNG_free(rng);
  
  // This is the selected states wanted at the end of the simulation
  SEXP r_ret = PROTECT(allocVector(REALSXP, obj->n_index_y));
  particle_copy_state(obj, REAL(r_ret));
  UNPROTECT(1);
  return r_ret;
}

SEXP r_particle_state(SEXP r_ptr) {
  particle *obj = (particle*) read_r_pointer(r_ptr, true);

  SEXP r_y = PROTECT(allocVector(REALSXP, obj->n_y));
  double *y = REAL(r_y);

  memcpy(y, obj->y, obj->n_y * sizeof(double));

  UNPROTECT(1);
  return(r_y);
}

SEXP r_particle_step(SEXP r_ptr) {
  particle *obj = (particle*) read_r_pointer(r_ptr, true);
  size_t step = obj->step;
  return ScalarInteger(step);
}

SEXP r_dust_alloc(SEXP r_create, SEXP r_update, SEXP r_free,
                  SEXP r_n_particles, SEXP r_nthreads, SEXP r_seed, 
                  SEXP r_y, SEXP user, SEXP r_index_y) {
  model_create * f_create = (model_create*) ptr_fn_get(r_create);
  model_update * f_update = (model_update*) ptr_fn_get(r_update);
  model_free * f_free = (model_free*) ptr_fn_get(r_free);

  size_t n_index_y = length(r_index_y);
  size_t *index_y = (size_t*) R_alloc(n_index_y, sizeof(size_t));
  for (size_t i = 0; i < n_index_y; ++i) {
    index_y[i] = INTEGER(r_index_y)[i] - 1;
  }
  size_t n_y = length(r_y);
  size_t n_particles = INTEGER(r_n_particles)[0];
  size_t nthreads = (size_t) INTEGER(r_nthreads)[0];
  uint64_t seed = (uint64_t) INTEGER(r_seed)[0];
  dust * obj = dust_alloc(f_create, f_update, f_free,
                          n_particles, nthreads, seed, 
                          n_y, REAL(r_y), user,
                          n_index_y, index_y);
  SEXP r_ptr = PROTECT(R_MakeExternalPtr(obj, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(r_ptr, dust_finalise);
  UNPROTECT(1);
  return r_ptr;
}

void dust_finalise(SEXP r_ptr) {
  dust *obj = (dust*) read_r_pointer(r_ptr, 0);
  if (r_ptr) {
    dust_free(obj);
    R_ClearExternalPtr(r_ptr);
  }
}

SEXP r_dust_run(SEXP r_ptr, SEXP r_step_end) {
  dust *obj = (dust*) read_r_pointer(r_ptr, true);
  size_t step_end = (size_t) INTEGER(r_step_end)[0];

  dust_run(obj, step_end);

  SEXP r_ret = PROTECT(allocMatrix(REALSXP, obj->n_index_y, obj->n_particles));
  dust_copy_state(obj, REAL(r_ret));
  UNPROTECT(1);
  return r_ret;
}

// pull the state out of the object (whole state here as the index
// we'll deal with elsewhere - probably putting it into the object?)
SEXP r_dust_state(SEXP r_ptr) {
  dust *obj = (dust*) read_r_pointer(r_ptr, true);

  size_t n_y = obj->n_y;
  SEXP r_y = PROTECT(allocMatrix(REALSXP, n_y, obj->n_particles));
  double *y = REAL(r_y);

  for (size_t i = 0; i < obj->n_particles; ++i) {
    particle * x = obj->particles + i;
    memcpy(y, x->y, n_y * sizeof(double));
    y += n_y;
  }

  UNPROTECT(1);
  return(r_y);
}

SEXP r_dust_step(SEXP r_ptr) {
  dust *obj = (dust*) read_r_pointer(r_ptr, true);
  size_t step = obj->particles[0].step;
  return ScalarInteger(step);
}

void* read_r_pointer(SEXP r_ptr, int closed_error) {
  void *ptr = NULL;
  if (TYPEOF(r_ptr) != EXTPTRSXP) {
    Rf_error("Expected an external pointer");
  }
  ptr = (void*) R_ExternalPtrAddr(r_ptr);
  if (!ptr && closed_error) {
    Rf_error("Pointer has been invalidated (perhaps serialised?)");
  }
  return ptr;
}

// This gets a function pointer from a data pointer and avoids some
// compiler warnings.
DL_FUNC ptr_fn_get(SEXP r_ptr) {
#if defined(R_VERSION) && R_VERSION >= R_Version(3, 4, 0)
  DL_FUNC ret = R_ExternalPtrAddrFn(r_ptr);
#else
  DL_FUNC ret = (DL_FUNC) R_ExternalPtrAddr(r_ptr);
#endif
  if (ret == NULL) {
    Rf_error("Function pointer has been invalidated (perhaps serialised?)");
  }
  return ret;
}
