// Generated by cpp11: do not edit by hand


#include "cpp11/declarations.hpp"

// dust.cpp
SEXP dust_volatility_alloc(cpp11::list r_data, size_t step, size_t n_particles, size_t n_threads, size_t seed);
extern "C" SEXP _volatility5f8227e8_dust_volatility_alloc(SEXP r_data, SEXP step, SEXP n_particles, SEXP n_threads, SEXP seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_alloc(cpp11::unmove(cpp11::as_cpp<cpp11::list>(r_data)), cpp11::unmove(cpp11::as_cpp<size_t>(step)), cpp11::unmove(cpp11::as_cpp<size_t>(n_particles)), cpp11::unmove(cpp11::as_cpp<size_t>(n_threads)), cpp11::unmove(cpp11::as_cpp<size_t>(seed))));
  END_CPP11
}
// dust.cpp
SEXP dust_volatility_run(SEXP ptr, size_t step_end);
extern "C" SEXP _volatility5f8227e8_dust_volatility_run(SEXP ptr, SEXP step_end) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_run(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<size_t>(step_end))));
  END_CPP11
}
// dust.cpp
SEXP dust_volatility_set_index(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _volatility5f8227e8_dust_volatility_set_index(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_set_index(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<cpp11::sexp>(r_index))));
  END_CPP11
}
// dust.cpp
SEXP dust_volatility_set_state(SEXP ptr, SEXP r_state, SEXP r_step);
extern "C" SEXP _volatility5f8227e8_dust_volatility_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_set_state(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<SEXP>(r_state)), cpp11::unmove(cpp11::as_cpp<SEXP>(r_step))));
  END_CPP11
}
// dust.cpp
SEXP dust_volatility_reset(SEXP ptr, cpp11::list r_data, size_t step);
extern "C" SEXP _volatility5f8227e8_dust_volatility_reset(SEXP ptr, SEXP r_data, SEXP step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_reset(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<cpp11::list>(r_data)), cpp11::unmove(cpp11::as_cpp<size_t>(step))));
  END_CPP11
}
// dust.cpp
SEXP dust_volatility_state(SEXP ptr, SEXP r_index);
extern "C" SEXP _volatility5f8227e8_dust_volatility_state(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_state(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<SEXP>(r_index))));
  END_CPP11
}
// dust.cpp
size_t dust_volatility_step(SEXP ptr);
extern "C" SEXP _volatility5f8227e8_dust_volatility_step(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_step(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr))));
  END_CPP11
}
// dust.cpp
void dust_volatility_reorder(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _volatility5f8227e8_dust_volatility_reorder(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    dust_volatility_reorder(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<cpp11::sexp>(r_index)));
    return R_NilValue;
  END_CPP11
}
// dust.cpp
SEXP dust_volatility_rng_state(SEXP ptr);
extern "C" SEXP _volatility5f8227e8_dust_volatility_rng_state(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_volatility_rng_state(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr))));
  END_CPP11
}

extern "C" {
/* .Call calls */
extern SEXP _volatility5f8227e8_dust_volatility_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_reorder(SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_reset(SEXP, SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_rng_state(SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_run(SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_set_index(SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_set_state(SEXP, SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_state(SEXP, SEXP);
extern SEXP _volatility5f8227e8_dust_volatility_step(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_volatility5f8227e8_dust_volatility_alloc",     (DL_FUNC) &_volatility5f8227e8_dust_volatility_alloc,     5},
    {"_volatility5f8227e8_dust_volatility_reorder",   (DL_FUNC) &_volatility5f8227e8_dust_volatility_reorder,   2},
    {"_volatility5f8227e8_dust_volatility_reset",     (DL_FUNC) &_volatility5f8227e8_dust_volatility_reset,     3},
    {"_volatility5f8227e8_dust_volatility_rng_state", (DL_FUNC) &_volatility5f8227e8_dust_volatility_rng_state, 1},
    {"_volatility5f8227e8_dust_volatility_run",       (DL_FUNC) &_volatility5f8227e8_dust_volatility_run,       2},
    {"_volatility5f8227e8_dust_volatility_set_index", (DL_FUNC) &_volatility5f8227e8_dust_volatility_set_index, 2},
    {"_volatility5f8227e8_dust_volatility_set_state", (DL_FUNC) &_volatility5f8227e8_dust_volatility_set_state, 3},
    {"_volatility5f8227e8_dust_volatility_state",     (DL_FUNC) &_volatility5f8227e8_dust_volatility_state,     2},
    {"_volatility5f8227e8_dust_volatility_step",      (DL_FUNC) &_volatility5f8227e8_dust_volatility_step,      1},
    {NULL, NULL, 0}
};
}

extern "C" void R_init_volatility5f8227e8(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
