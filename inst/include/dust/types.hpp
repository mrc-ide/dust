#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

#include <memory>

namespace dust {

struct nothing {};
typedef nothing no_data;
typedef nothing no_internal;
typedef nothing no_shared;

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
    pars_type(shared_, dust::nothing()) {
  }
  pars_type(typename T::internal_type internal_) :
    pars_type(dust::nothing(), internal_) {
  }
};

}

#endif
