#include <R.h>
#include <Rinternals.h>
#include <Rversion.h>

#include "dust.h"

SEXP r_particle_alloc(SEXP r_create, SEXP r_update, SEXP r_free,
                      SEXP r_y, SEXP user, SEXP r_index_y);
SEXP r_particle_run(SEXP r_ptr, SEXP r_step_end, SEXP r_seed);
SEXP r_particle_state(SEXP r_ptr);
SEXP r_particle_step(SEXP r_ptr);

SEXP r_dust_alloc(SEXP r_create, SEXP r_update, SEXP r_free,
                  SEXP r_n_particles, SEXP r_nthreads, SEXP r_model_rng_calls, 
                  SEXP r_y, SEXP user, SEXP r_index_y);
SEXP r_dust_run(SEXP r_ptr, SEXP r_step_end);
SEXP r_dust_state(SEXP r_ptr);
SEXP r_dust_step(SEXP r_ptr);

void particle_finalise(SEXP r_ptr);
void dust_finalise(SEXP r_ptr);
void* read_r_pointer(SEXP r_ptr, int closed_error);
DL_FUNC ptr_fn_get(SEXP r_ptr);

SEXP r_binom_test(SEXP r_type);