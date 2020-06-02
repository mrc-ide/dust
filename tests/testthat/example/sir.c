#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <stdbool.h>
#include <R_ext/Rdynload.h>

typedef struct sir_internal {
  double beta;
  double dt;
  double gamma;
  double I0;
  double initial_I;
  double initial_R;
  double initial_S;
  double p_IR;
  double S0;
  double steps_per_day;
} sir_internal;

sir_internal* sir_get_internal(SEXP internal_p, int closed_error);
SEXP sir_create(SEXP user);

SEXP sir_initial_conditions(SEXP internal_p, SEXP step_ptr);
SEXP sir_set_user(SEXP internal_p, SEXP user);
void sir_rhs(sir_internal* internal, size_t step, const double * state, double * state_next, double * output);
void sir_rhs_dde(size_t n_eq, size_t step, const double * state, double * state_next, size_t n_out, double * output, const void * internal);

// Functions for the ffi
double user_get_scalar_double(SEXP user, const char *name,
                              double default_value, double min, double max);
int user_get_scalar_int(SEXP user, const char *name,
                        int default_value, double min, double max);
void user_check_values_double(double * value, size_t len,
                                  double min, double max, const char *name);
void user_check_values_int(int * value, size_t len,
                               double min, double max, const char *name);
void user_check_values(SEXP value, double min, double max,
                           const char *name);
SEXP user_list_element(SEXP list, const char *name);

// Added
void * sir2_create(SEXP user);
void sir2_free(void * data);
void sir2_set_user(sir_internal * internal, SEXP user);
void sir2_update(void* data, size_t step, const double * state,
                 double * state_next);


sir_internal* sir_get_internal(SEXP internal_p, int closed_error) {
  sir_internal *internal = NULL;
  if (TYPEOF(internal_p) != EXTPTRSXP) {
    Rf_error("Expected an external pointer");
  }
  internal = (sir_internal*) R_ExternalPtrAddr(internal_p);
  if (!internal && closed_error) {
    Rf_error("Pointer has been invalidated");
  }
  return internal;
}

void sir_finalise(SEXP internal_p) {
  sir_internal *internal = sir_get_internal(internal_p, 0);
  if (internal_p) {
    Free(internal);
    R_ClearExternalPtr(internal_p);
  }
}

SEXP sir_create(SEXP user) {
  sir_internal *internal = (sir_internal*) Calloc(1, sir_internal);
  internal->initial_R = 0;
  internal->beta = 0.20000000000000001;
  internal->gamma = 0.10000000000000001;
  internal->I0 = 10;
  internal->S0 = 1000;
  internal->steps_per_day = 4;
  SEXP ptr = PROTECT(R_MakeExternalPtr(internal, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(ptr, sir_finalise);
  UNPROTECT(1);
  return ptr;
}

SEXP sir_set_user(SEXP internal_p, SEXP user) {
  sir_internal *internal = sir_get_internal(internal_p, 1);
  internal->beta = user_get_scalar_double(user, "beta", internal->beta, NA_REAL, NA_REAL);
  internal->gamma = user_get_scalar_double(user, "gamma", internal->gamma, NA_REAL, NA_REAL);
  internal->I0 = user_get_scalar_double(user, "I0", internal->I0, NA_REAL, NA_REAL);
  internal->S0 = user_get_scalar_double(user, "S0", internal->S0, NA_REAL, NA_REAL);
  internal->steps_per_day = user_get_scalar_double(user, "steps_per_day", internal->steps_per_day, NA_REAL, NA_REAL);
  internal->dt = 1 / (double) internal->steps_per_day;
  internal->initial_I = internal->I0;
  internal->initial_S = internal->S0;
  internal->p_IR = 1 - exp(-(internal->gamma));
  return R_NilValue;
}

SEXP sir_initial_conditions(SEXP internal_p, SEXP step_ptr) {
  sir_internal *internal = sir_get_internal(internal_p, 1);
  SEXP r_state = PROTECT(allocVector(REALSXP, 3));
  double * state = REAL(r_state);
  state[0] = internal->initial_S;
  state[1] = internal->initial_I;
  state[2] = internal->initial_R;
  UNPROTECT(1);
  return r_state;
}

void sir_rhs(sir_internal* internal, size_t step, const double * state, double * state_next, double * output) {
  double S = state[0];
  double I = state[1];
  double R = state[2];
  double N = S + I + R;
  double n_IR = Rf_rbinom(round(I), internal->p_IR * internal->dt);
  double p_SI = 1 - exp(-(internal->beta) * I / (double) N);
  double n_SI = Rf_rbinom(round(S), p_SI * internal->dt);
  state_next[2] = R + n_IR;
  state_next[1] = I + n_SI - n_IR;
  state_next[0] = S - n_SI;
  output[1] = step * internal->dt;
  output[0] = n_SI;
}

void sir_rhs_dde(size_t n_eq, size_t step, const double * state, double * state_next, size_t n_out, double * output, const void * internal) {
  sir_rhs((sir_internal*)internal, step, state, state_next, output);
}

// Functions used to help with the FFI
double user_get_scalar_double(SEXP user, const char *name,
                              double default_value, double min, double max) {
  double ret = default_value;
  SEXP el = user_list_element(user, name);
  if (el != R_NilValue) {
    if (length(el) != 1) {
      Rf_error("Expected a scalar numeric for '%s'", name);
    }
    if (TYPEOF(el) == REALSXP) {
      ret = REAL(el)[0];
    } else if (TYPEOF(el) == INTSXP) {
      ret = INTEGER(el)[0];
    } else {
      Rf_error("Expected a numeric value for %s", name);
    }
  }
  if (ISNA(ret)) {
    Rf_error("Expected a value for '%s'", name);
  }
  user_check_values_double(&ret, 1, min, max, name);
  return ret;
}

int user_get_scalar_int(SEXP user, const char *name,
                        int default_value, double min, double max) {
  int ret = default_value;
  SEXP el = user_list_element(user, name);
  if (el != R_NilValue) {
    if (length(el) != 1) {
      Rf_error("Expected scalar integer for %d", name);
    }
    if (TYPEOF(el) == REALSXP) {
      double tmp = REAL(el)[0];
      if (fabs(tmp - round(tmp)) > 2e-8) {
        Rf_error("Expected '%s' to be integer-like", name);
      }
    }
    ret = INTEGER(coerceVector(el, INTSXP))[0];
  }
  if (ret == NA_INTEGER) {
    Rf_error("Expected a value for '%s'", name);
  }
  user_check_values_int(&ret, 1, min, max, name);
  return ret;
}

void user_check_values_double(double * value, size_t len,
                                  double min, double max, const char *name) {
  for (size_t i = 0; i < len; ++i) {
    if (ISNA(value[i])) {
      Rf_error("'%s' must not contain any NA values", name);
    }
  }
  if (min != NA_REAL) {
    for (size_t i = 0; i < len; ++i) {
      if (value[i] < min) {
        Rf_error("Expected '%s' to be at least %g", name, min);
      }
    }
  }
  if (max != NA_REAL) {
    for (size_t i = 0; i < len; ++i) {
      if (value[i] > max) {
        Rf_error("Expected '%s' to be at most %g", name, max);
      }
    }
  }
}

void user_check_values_int(int * value, size_t len,
                               double min, double max, const char *name) {
  for (size_t i = 0; i < len; ++i) {
    if (ISNA(value[i])) {
      Rf_error("'%s' must not contain any NA values", name);
    }
  }
  if (min != NA_REAL) {
    for (size_t i = 0; i < len; ++i) {
      if (value[i] < min) {
        Rf_error("Expected '%s' to be at least %g", name, min);
      }
    }
  }
  if (max != NA_REAL) {
    for (size_t i = 0; i < len; ++i) {
      if (value[i] > max) {
        Rf_error("Expected '%s' to be at most %g", name, max);
      }
    }
  }
}

void user_check_values(SEXP value, double min, double max,
                           const char *name) {
  size_t len = (size_t)length(value);
  if (TYPEOF(value) == INTSXP) {
    user_check_values_int(INTEGER(value), len, min, max, name);
  } else {
    user_check_values_double(REAL(value), len, min, max, name);
  }
}

SEXP user_list_element(SEXP list, const char *name) {
  SEXP ret = R_NilValue, names = getAttrib(list, R_NamesSymbol);
  for (int i = 0; i < length(list); ++i) {
    if(strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
      ret = VECTOR_ELT(list, i);
      break;
    }
  }
  return ret;
}

void sir2_set_user(sir_internal * internal, SEXP user) {
  internal->beta = user_get_scalar_double(user, "beta", internal->beta, NA_REAL, NA_REAL);
  internal->gamma = user_get_scalar_double(user, "gamma", internal->gamma, NA_REAL, NA_REAL);
  internal->I0 = user_get_scalar_double(user, "I0", internal->I0, NA_REAL, NA_REAL);
  internal->S0 = user_get_scalar_double(user, "S0", internal->S0, NA_REAL, NA_REAL);
  internal->steps_per_day = user_get_scalar_double(user, "steps_per_day", internal->steps_per_day, NA_REAL, NA_REAL);
  internal->dt = 1 / (double) internal->steps_per_day;
  internal->initial_I = internal->I0;
  internal->initial_S = internal->S0;
  internal->p_IR = 1 - exp(-(internal->gamma));
}

void * sir2_create(SEXP user) {
  sir_internal *internal = (sir_internal*) Calloc(1, sir_internal);
  internal->initial_R = 0;
  internal->beta = 0.20000000000000001;
  internal->gamma = 0.10000000000000001;
  internal->I0 = 10;
  internal->S0 = 1000;
  internal->steps_per_day = 4;
  sir2_set_user(internal, user);
  return internal;
}

void sir2_free(void * data) {
  sir_internal * internal = (sir_internal*) data;
  if (internal) {
    // Clear any created data
    Free(internal);
  }
}

void sir2_update(void* data, size_t step, const double * state,
                 double * state_next) {
  sir_internal * internal = (sir_internal*)data;
  double S = state[0];
  double I = state[1];
  double R = state[2];
  double N = S + I + R;
  double n_IR = Rf_rbinom(round(I), internal->p_IR * internal->dt);
  double p_SI = 1 - exp(-(internal->beta) * I / (double) N);
  double n_SI = Rf_rbinom(round(S), p_SI * internal->dt);
  state_next[2] = R + n_IR;
  state_next[1] = I + n_SI - n_IR;
  state_next[0] = S - n_SI;
}
