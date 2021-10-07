#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

namespace dust {

struct nothing {};
typedef nothing no_data;
typedef nothing no_internal;
typedef nothing no_shared;

// By default we do not support anything on the gpu. This name might
// change, but it does reflect our intent and it's likely that to work
// on a GPU the model will have to provide a number of things. If of
// those becomes a type (as with data, internal and shared) we could
// use the same approach as above.
template <typename T>
struct has_gpu_support : std::false_type {};

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
