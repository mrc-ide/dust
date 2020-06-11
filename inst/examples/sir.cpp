// NOTE: Before moving to odin:
//
// * we will probably need a data struct to avoid name clashes with the object
// * also need a save way of throwing with try/catch etc and *not* Rf_error during parameter checks

// Utilities based on odin
double user_get_scalar_double(SEXP user, const char *name,
                              double default_value, double min, double max);
SEXP user_list_element(SEXP list, const char *name);

class sir {
public:
  typedef SEXP init_t;
  sir(SEXP data) :
    initial_R(0),
    beta(0.2),
    gamma(0.1),
    I0(10),
    S0(1000),
    steps_per_day(4) {
    set_user(data);
  }

  size_t size() const {
    return 3;
  }

  std::vector<double> initial(size_t step) {
    std::vector<double> ret = {initial_S, initial_I, initial_R};
    return ret;
  }

  void update(size_t step, const std::vector<double> state, dust::RNG& rng,
              std::vector<double>& state_next) {
    double S = state[0];
    double I = state[1];
    double R = state[2];
    double N = S + I + R;
    double n_IR = rng.rbinom(round(I), p_IR * dt);
    double p_SI = 1 - exp(-(beta) * I / (double) N);
    double n_SI = rng.rbinom(round(S), p_SI * dt);
    state_next[2] = R + n_IR;
    state_next[1] = I + n_SI - n_IR;
    state_next[0] = S - n_SI;
  }

private:
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

  void set_user(SEXP data);
};


void sir::set_user(SEXP user) {
  beta = user_get_scalar_double(user, "beta", beta, NA_REAL, NA_REAL);
  gamma = user_get_scalar_double(user, "gamma", gamma, NA_REAL, NA_REAL);
  I0 = user_get_scalar_double(user, "I0", I0, NA_REAL, NA_REAL);
  S0 = user_get_scalar_double(user, "S0", S0, NA_REAL, NA_REAL);
  steps_per_day = user_get_scalar_double(user, "steps_per_day", steps_per_day,
                                         NA_REAL, NA_REAL);
  dt = 1 / (double) steps_per_day;
  initial_I = I0;
  initial_S = S0;
  p_IR = 1 - exp(-(gamma));
}


// Copied in from odin - not generally safe before of error handling
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
  return ret;
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
