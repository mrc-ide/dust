#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <Rversion.h>

#include "test_rng.h"
#include "example_walk.h"

static const R_CallMethodDef call_methods[] = {
  {"Ctest_rng",      (DL_FUNC) &test_rng,        2},
  {"Ctest_walk",     (DL_FUNC) &test_walk,       3},

  {NULL,              NULL,                      0}
};

void R_init_dust(DllInfo *info) {
  // Register C routines to be called from R:
  R_registerRoutines(info, NULL, call_methods, NULL, NULL);

#if defined(R_VERSION) && R_VERSION >= R_Version(3, 3, 0)
  R_useDynamicSymbols(info, FALSE);
  R_forceSymbols(info, TRUE);
#endif
}
