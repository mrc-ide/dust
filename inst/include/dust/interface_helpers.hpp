#ifndef DUST_INTERFACE_HELPERS_HPP
#define DUST_INTERFACE_HELPERS_HPP

namespace dust {
namespace helpers {

cpp11::writable::integers state_array_dim(size_t n_state,
                                          const std::vector<size_t>& shape) {
  cpp11::writable::integers dim(shape.size() + 1);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  return dim;
}

cpp11::writable::integers state_array_dim(size_t n_state,
                                          const std::vector<size_t>& shape,
                                          size_t n_time) {
  cpp11::writable::integers dim(shape.size() + 2);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  dim[dim.size() - 1] = n_time;
  return dim;
}

cpp11::sexp state_array(const std::vector<double>& dat, size_t n_state,
                        const std::vector<size_t>& shape) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = state_array_dim(n_state, shape);

  return ret;
}

cpp11::sexp state_array(const std::vector<double>& dat, size_t n_state,
                        const std::vector<size_t>& shape, size_t n_time) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  ret.attr("dim") = state_array_dim(n_state, shape, n_time);

  return ret;
}

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
      cpp11::stop("Expected a list with %d elements for 'pars' but given %d",
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
      cpp11::stop("Expected dimension %d of 'state' to be %d but given %d",
                  i + 1, expected, found);
    }
  }

  const size_t len = r_state_data.size();
  std::vector<real_t> ret(len);
  std::copy_n(REAL(r_state_data.data()), len, ret.begin());
  return ret;
}


template <typename T>
T product(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

}
}

#endif
