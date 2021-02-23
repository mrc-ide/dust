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
                    i, expected, found);
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

}
}

#endif
