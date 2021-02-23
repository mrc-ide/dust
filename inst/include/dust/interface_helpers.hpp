#ifndef DUST_INTERFACE_HELPERS_HPP
#define DUST_INTERFACE_HELPERS_HPP

namespace dust {
namespace helpers {

cpp11::sexp state_array(const std::vector<double>& dat, size_t n_state,
                        const std::vector<size_t>& shape) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  cpp11::writable::integers dim(shape.size() + 1);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  ret.attr("dim") = dim;

  return ret;
}

cpp11::sexp state_array(const std::vector<double>& dat, size_t n_state,
                        const std::vector<size_t>& shape, size_t n_time) {
  cpp11::writable::doubles ret(dat.size());
  std::copy(dat.begin(), dat.end(), REAL(ret));

  cpp11::writable::integers dim(shape.size() + 2);
  dim[0] = n_state;
  std::copy(shape.begin(), shape.end(), dim.begin() + 1);
  dim[dim.size() - 1] = n_time;
  ret.attr("dim") = dim;

  return ret;
}

}
}

#endif
