#ifndef DUST_TYPES_HPP
#define DUST_TYPES_HPP

#include <memory>

namespace dust {

struct no_data {};
struct no_internal {};
struct no_shared {};
struct no_constant {};

template <typename T>
using constant_ptr = std::shared_ptr<const typename T::constant_type>;

template <typename T>
using shared_ptr = std::shared_ptr<const typename T::shared_type>;

// There are three types of information that are used when defining this:
//
// * internal - this is the model's internal data structures, holding
//   information that is required per particle in order to run the
//   calculations but which is never referred to directly (i.e.,
//   scratch space). The size of this is fixed as the parameters are
//   created.
//
// * constant - these are parameters that are shared across all uses of a
//   parameter set. This should include all size parameters
//
// * shared - these are parameters that can vary between constructions
//   of parameter objects, but are shared between diferent particles.
template <typename T>
struct pars_type {
  std::shared_ptr<const typename T::shared_type> shared;
  std::shared_ptr<const typename T::constant_type> constant;
  typename T::internal_type internal;

  pars_type(std::shared_ptr<const typename T::shared_type> shared_,
            std::shared_ptr<const typename T::constant_type> constant_,
            typename T::internal_type internal_) :
    shared(std::make_shared<const typename T::shared_type>(shared_)),
    constant(std::make_shared<const typename T::constant_type>(constant_)),
    internal(internal_) {
  }
};

}

#endif
