#ifndef DUST_INTERFACE_UTILS_HPP
#define DUST_INTERFACE_UTILS_HPP

#include <cpp11.hpp>

#include "dust/types.hpp"

namespace dust {

template <typename T>
typename dust::pars_type<T> dust_pars(cpp11::list pars);

template <typename T>
typename T::data_type dust_data(cpp11::list data);

template <typename T>
cpp11::sexp dust_info(const dust::pars_type<T>& pars) {
  return R_NilValue;
}

}

#endif
