#ifndef DUST_INTERFACE_HELPERS_HPP
#define DUST_INTERFACE_HELPERS_HPP

namespace dust {
namespace interface {

inline
cpp11::integers as_integer(cpp11::sexp x, const char * name) {
  if (TYPEOF(x) == INTSXP) {
    return cpp11::as_cpp<cpp11::integers>(x);
  } else if (TYPEOF(x) == REALSXP) {
    cpp11::doubles xn = cpp11::as_cpp<cpp11::doubles>(x);
    size_t len = xn.size();
    cpp11::writable::integers ret = cpp11::writable::integers(len);
    for (size_t i = 0; i < len; ++i) {
      double el = xn[i];
      if (!cpp11::is_convertable_without_loss_to_integer(el)) {
        cpp11::stop("All elements of '%s' must be integer-like",
                    name, i + 1);
      }
      ret[i] = static_cast<int>(el);
    }
    return ret;
  } else {
    cpp11::stop("Expected a numeric vector for '%s'", name);
    return cpp11::integers(); // never reached
  }
}

inline
void validate_size(int x, const char * name) {
  if (x < 0) {
    cpp11::stop("'%s' must be non-negative", name);
  }
}

inline
void validate_positive(int x, const char *name) {
  if (x <= 0) {
    cpp11::stop("'%s' must be positive", name);
  }
}

inline
std::vector<size_t> validate_size(cpp11::sexp r_x, const char * name) {
  cpp11::integers r_xi = as_integer(r_x, name);
  const size_t n = static_cast<size_t>(r_xi.size());
  std::vector<size_t> x;
  x.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int el = r_xi[i];
    if (el < 0) {
      cpp11::stop("All elements of '%s' must be non-negative", name);
    }
    x.push_back(el);
  }
  return x;
}

// Helper for the below; in the case where index is not given we
// assume it would have been given as 1..n so generate out 0..(n-1)
inline
std::vector<size_t> seq_len(size_t n) {
  std::vector<size_t> index;
  index.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    index.push_back(i);
  }
  return index;
}

// Converts an R vector of integers (in base-1) to a C++ std::vector
// of size_t values in base-0 having checked that the values of the
// vectors are approproate; that they will not fall outside of the
// range [1, nmax] in base-1.
inline
std::vector<size_t> r_index_to_index(cpp11::sexp r_index, size_t nmax) {
  if (r_index == R_NilValue) {
    return seq_len(nmax);
  }

  cpp11::integers r_index_int = as_integer(r_index, "index");
  const int n = r_index_int.size();
  std::vector<size_t> index;
  index.reserve(n);
  for (int i = 0; i < n; ++i) {
    int x = r_index_int[i];
    if (x < 1 || x > (int)nmax) {
      cpp11::stop("All elements of 'index' must lie in [1, %d]", nmax);
    }
    index.push_back(x - 1);
  }
  return index;
}

inline
void check_dimensions(cpp11::sexp obj, size_t obj_size,
                      const std::vector<size_t>& shape,
                      const char * name) {
  cpp11::integers dim;
  auto r_dim = obj.attr("dim");
  if (r_dim == R_NilValue) {
    dim = cpp11::writable::integers{static_cast<int>(obj_size)};
  } else {
    dim = cpp11::as_cpp<cpp11::integers>(r_dim);
  }

  const size_t dim_len = dim.size();
  if (dim_len != shape.size()) {
    if (shape.size() == 1) {
      cpp11::stop("Expected a vector for '%s'", name);
    } else {
      cpp11::stop("Expected an array of rank %d for '%s'",
                  shape.size(), name);
    }
  }

  for (size_t i = 0; i < shape.size(); ++i) {
    const size_t found = dim[i], expected = shape[i];
    if (found != expected) {
      if (shape.size() == 1) {
        cpp11::stop("Expected a vector of length %d for '%s' but given %d",
                    expected, name, found);
      } else {
        cpp11::stop("Expected dimension %d of '%s' to be %d but given %d",
                    i + 1, name, expected, found);
      }
    }
  }
}

inline
cpp11::writable::integers state_array_dim(size_t n_state,
                                          const std::vector<size_t>& shape) {
  cpp11::writable::integers dim(shape.size() + 1);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  return dim;
}

inline
cpp11::writable::integers state_array_dim(size_t n_state,
                                          const std::vector<size_t>& shape,
                                          size_t n_time) {
  cpp11::writable::integers dim(shape.size() + 2);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  dim[dim.size() - 1] = n_time;
  return dim;
}

template <typename real_t>
cpp11::sexp state_array(const std::vector<real_t>& dat, size_t n_state,
                        const std::vector<size_t>& shape) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = state_array_dim(n_state, shape);

  return ret;
}

template <typename real_t>
cpp11::sexp state_array(const std::vector<real_t>& dat, size_t n_state,
                        const std::vector<size_t>& shape, size_t n_time) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = state_array_dim(n_state, shape, n_time);

  return ret;
}

inline
cpp11::writable::integers vector_size_to_int(const std::vector<size_t> & x) {
  cpp11::writable::integers ret(x.size());
  std::copy(x.begin(), x.end(), ret.begin());
  return ret;
}

inline void check_pars_multi(cpp11::list r_pars,
                             const std::vector<size_t>& shape) {
  if (r_pars.attr("names") != R_NilValue) {
    cpp11::stop("Expected an unnamed list for 'pars' (given 'pars_multi')");
  }
  cpp11::sexp r_dim = r_pars.attr("dim");
  if (shape.size() <= 2) {
    const size_t expected = shape.back();
    if (r_dim != R_NilValue) {
      cpp11::stop("Expected a list with no dimension attribute for 'pars'");
    }
    if (static_cast<size_t>(r_pars.size()) != expected) {
      cpp11::stop("Expected a list of length %d for 'pars' but given %d",
                  expected, r_pars.size());
    }
  } else {
    if (r_dim == R_NilValue) {
      cpp11::stop("Expected a list with a dimension attribute for 'pars'");
    }
    cpp11::integers r_dim_int = cpp11::as_cpp<cpp11::integers>(r_dim_int);
    if (static_cast<size_t>(r_dim_int.size()) != shape.size()) {
      cpp11::stop("Expected 'pars' to have rank %d but given rank %d",
                  shape.size(), r_dim_int.size());
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      const size_t expected = shape[i], found = r_dim_int[i];
      if (found != expected) {
        cpp11::stop("Expected dimension %d of 'pars' to be %d but given %d",
                    i + 1, expected, found);
      }
    }
  }
}

inline void check_pars_multi(cpp11::list r_pars) {
  if (r_pars.attr("names") != R_NilValue) {
    cpp11::stop("Expected an unnamed list for 'pars' (given 'pars_multi')");
  }
  if (r_pars.size() == 0) {
    cpp11::stop("Expected 'pars' to have at least one element");
  }
}

// TODO: consider behaviour of the recycling here (if given a vector
// of n_particles). Can this be extended to multiparameter sets
// reasonably? I guess the logic would be that *exactly* the
// n_particles dimension is missing? The other option is to set *all*
// to the same value.
//
// no pars: shape = {n_particles}
//   vector of length n_state => shared
//   matrix of n_state x n_particles => individual
//
// with pars: shape = {n_particles_each, n_pars}
//   matrix of n_state x n_pars => shared per parameter
//   array of n_state x n_particles_each x n_pars => individual
//
// with pars: shape = {n_particles_each, n_a, n_b}
//   matrix of n_state x n_a x n_b => shared per parameter
//   array of n_state x n_particles_each x n_a x n_b => individual
template <typename real_t>
std::vector<real_t> check_state(cpp11::sexp r_state, size_t n_state,
                                const std::vector<size_t>& shape) {
  cpp11::doubles r_state_data = cpp11::as_cpp<cpp11::doubles>(r_state);
  auto r_dim = r_state.attr("dim");
  cpp11::integers dim;
  if (r_dim == R_NilValue) {
    dim = cpp11::writable::integers{static_cast<int>(r_state_data.size())};
  } else {
    dim = cpp11::as_cpp<cpp11::integers>(r_dim);
  }
  const size_t dim_len = dim.size();

  // Expected lengths
  const size_t len_shared = shape.size(), len_individual = shape.size() + 1;

  if (dim_len != len_shared && dim_len != len_individual) {
    cpp11::stop("Expected array of rank %d or %d for 'state' but given rank %d",
                len_shared, len_individual, dim_len);
  }

  const bool is_shared = dim_len == len_shared;
  for (size_t i = 0; i < static_cast<size_t>(dim_len); ++i) {
    const size_t found = dim[i];
    size_t expected;
    if (i == 0) {
      expected = n_state;
    } else {
      expected = is_shared ? shape[i] : shape[i - 1];
    }
    if (found != expected) {
      if (dim_len == 1) {
        cpp11::stop("Expected a vector of length %d for 'state' but given %d",
                    expected, found);
      } else {
        cpp11::stop("Expected dimension %d of 'state' to be %d but given %d",
                    i + 1, expected, found);
      }
    }
  }

  const size_t len = r_state_data.size();
  std::vector<real_t> ret(len);
  std::copy_n(REAL(r_state_data.data()), len, ret.begin());
  return ret;
}

// There are few options here, and this transform applies to other
// things (reorder weights)
//
// single: expect a vector of length shape[0] (n_particles)
//
// groups: expect a matrix of size n_particles x n_groups (this is shape)
inline
std::vector<size_t> check_reorder_index(cpp11::sexp r_index,
                                        const std::vector<size_t>& shape) {
  cpp11::integers r_index_data = as_integer(r_index, "index");
  const size_t len = r_index_data.size();
  check_dimensions(r_index, len, shape, "index");

  const size_t n_particles = shape[0];
  const size_t n_groups = len  / n_particles;
  std::vector<size_t> index;
  index.reserve(len);
  for (size_t i = 0, j = 0; i < n_groups; ++i) {
    for (size_t k = 0; k < n_particles; ++j, ++k) {
      int x = r_index_data[j];
      if (x < 1 || x > (int)n_particles) {
        cpp11::stop("All elements of 'index' must lie in [1, %d]", n_particles);
      }
      index.push_back(i * n_particles + x - 1);
    }
  }

  return index;
}

template <typename real_t>
std::vector<real_t> check_resample_weights(cpp11::doubles r_weights,
                                           const std::vector<size_t>& shape) {
  const size_t len = r_weights.size();
  check_dimensions(r_weights, len, shape, "weights");
  if (*std::min_element(r_weights.begin(), r_weights.end()) < 0) {
    cpp11::stop("All weights must be positive");
  }
  const std::vector<real_t>
    weights(r_weights.begin(), r_weights.end());
  return weights;
}

}
}

#endif
