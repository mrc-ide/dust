#ifndef DUST_UTIL_HPP
#define DUST_UTIL_HPP

#include <R.h>
#include <Rinternals.h>

namespace dust {
namespace util {
template <typename T>
T* read_r_pointer(SEXP r_ptr, bool closed_error) {
  void *ptr = NULL;
  if (TYPEOF(r_ptr) != EXTPTRSXP) {
    Rf_error("Expected an external pointer");
  }
  ptr = (void*) R_ExternalPtrAddr(r_ptr);
  if (!ptr && closed_error) {
    Rf_error("Pointer has been invalidated (perhaps serialised?)");
  }
  return static_cast<T*>(ptr);
}

inline int as_double(SEXP x, const char * name) {
  if (length(x) != 1) {
    Rf_error("Expected a scalar for '%s'", name);
  }
  double ret;
  if (TYPEOF(x) == INTSXP) {
    ret = INTEGER(x)[0];
  } else if (TYPEOF(x) == REALSXP) {
    ret = REAL(x)[0];
  } else {
    Rf_error("Expected a double for '%s'", name);
  }
  return ret;
}

inline int as_integer(SEXP x, const char * name) {
  if (length(x) != 1) {
    Rf_error("Expected a scalar for '%s'", name);
  }
  int ret;
  if (TYPEOF(x) == INTSXP) {
    ret = INTEGER(x)[0];
  } else if (TYPEOF(x) == REALSXP) {
    double value = REAL(x)[0];
    ret = value;
    if (ret != value) {
      Rf_error("Expected an integer for '%s' (rounding error?)", name);
    }
  } else {
    Rf_error("Expected an integer for '%s'", name);
  }
  return ret;
}

inline size_t as_size(SEXP x, const char * name) {
  int value = as_integer(x, name);
  if (value < 0) {
    Rf_error("Expected a non-negative integer for '%s'", name);
  }
  return static_cast<size_t>(value);
}

inline void validate_n(size_t n_generators, size_t n_threads) {
  if (n_generators < n_threads) {
    Rf_error("n_generators must be at least n_threads");
  }
  if (n_generators % n_threads > 0) {
    Rf_error("n_generators must be a multiple of n_threads");
  }
}

}
}

#endif
