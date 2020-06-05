#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <stdbool.h>
#include <R_ext/Rdynload.h>

double dust_rnorm(void* rng, size_t thread_idx, double mu, double sd) {
  typedef double dust_rnorm_t(void*, size_t, double, double);
  static dust_rnorm_t *fun;
  if (fun == NULL) {
    fun = (dust_rnorm_t*)
      R_GetCCallable("dust", "dust_rnorm");
  }
  return fun(rng, thread_idx, mu, sd);
}

typedef struct walk_internal {
  double mean;
  double sd;
} walk_internal;

void * walk_create(SEXP user);
void walk_free(void * data);
void walk_update(void* data, size_t step, const double * state,
                 void* rng, size_t thread_idx, double * state_next);

void * walk_create(SEXP user) {
  walk_internal *internal = (walk_internal*) Calloc(1, walk_internal);
  double * pars = REAL(user);
  internal->mean = pars[0];
  internal->sd = pars[1];
  return internal;
}

void walk_free(void * data) {
  walk_internal * internal = (walk_internal*) data;
  if (internal) {
    Free(internal);
  }
}

void walk_update(void* data, size_t step, const double * state,
                  void *rng, size_t thread_idx, double * state_next) {
  walk_internal * internal = (walk_internal*)data;
  double diff = dust_rnorm(rng, thread_idx, internal->mean, internal->sd);
  state_next[0] = state[0] + diff;
}
