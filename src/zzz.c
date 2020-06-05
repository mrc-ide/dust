#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <Rversion.h>

#include "interface.h"

static const R_CallMethodDef call_methods[] = {
  {"Cparticle_alloc",  (DL_FUNC) &r_particle_alloc,    6},
  {"Cparticle_run",    (DL_FUNC) &r_particle_run,      2},
  {"Cparticle_state",  (DL_FUNC) &r_particle_state,    1},
  {"Cparticle_step",   (DL_FUNC) &r_particle_step,     1},

  {"Cdust_alloc",      (DL_FUNC) &r_dust_alloc,        9},
  {"Cdust_run",        (DL_FUNC) &r_dust_run,          2},
  {"Cdust_state",      (DL_FUNC) &r_dust_state,        1},
  {"Cdust_step",       (DL_FUNC) &r_dust_step,         1},

  {NULL,              NULL,                            0}
};

void R_init_dust(DllInfo *info) {
  // Register C routines to be called from R:
  R_registerRoutines(info, NULL, call_methods, NULL, NULL);

  R_RegisterCCallable("dust", "dust_rbinom",
                      (DL_FUNC) &C_rbinom);
  R_RegisterCCallable("dust", "dust_rnorm",
                      (DL_FUNC) &C_rnorm);

#if defined(R_VERSION) && R_VERSION >= R_Version(3, 3, 0)
  R_useDynamicSymbols(info, FALSE);
  R_forceSymbols(info, TRUE);
#endif
}
