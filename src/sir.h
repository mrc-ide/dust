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
