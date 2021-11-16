#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

#include <memory>

namespace dust {

struct no_data {};
struct no_internal {};
struct no_shared {};

template <typename T>
using shared_ptr = std::shared_ptr<const typename T::shared_type>;

template <typename T>
struct pars_type {
  std::shared_ptr<const typename T::shared_type> shared;
  typename T::internal_type internal;

  pars_type(std::shared_ptr<const typename T::shared_type> shared_,
         typename T::internal_type internal_) :
    shared(shared_), internal(internal_) {
  }
  pars_type(typename T::shared_type shared_,
         typename T::internal_type internal_) :
    shared(std::make_shared<const typename T::shared_type>(shared_)),
    internal(internal_) {
  }
  pars_type(typename T::shared_type shared_) :
    pars_type(shared_, dust::no_internal()) {
  }
  pars_type(typename T::internal_type internal_) :
    pars_type(dust::no_shared(), internal_) {
  }
};

}

#endif
