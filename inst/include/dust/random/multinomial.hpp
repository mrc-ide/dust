#ifndef DUST_RANDOM_MULTINOMIAL_HPP
#define DUST_RANDOM_MULTINOMIAL_HPP

#include <cmath>
#include <dust/random/numeric.hpp>

namespace dust {
namespace random {

// should be able to write this pretty easily over container types but
// for now doing it explictly over pointers.
//
// Note that 'prob' and 'ret' (T) will ordinarily be real_type but
// that might not be the case, in particular when calling from R where
// we want calculations to happen on reals but the inputs (and
// destination) are double precision.
//
// TODO: A useful feature for later will be the concept of 'stride';
// with this we have one of the required bits for proper odin support.
template <typename real_type, typename rng_state_type, typename T>
void multinomial(rng_state_type& rng_state, int size, const T * prob,
                 int len, T * ret) {
  real_type p_tot = 0;
  for (int i = 0; i < len; ++i) {
    p_tot += prob[i];
  }

  for (int i = 0; i < len - 1; ++i) {
    if (prob[i] > 0) {
      const real_type pi = std::min(static_cast<real_type>(prob[i]) / p_tot,
                                    static_cast<real_type>(1));
      ret[i] = binomial<real_type>(rng_state, size, pi);
      size -= ret[i];
      p_tot -= prob[i];
    } else {
      ret[i] = 0;
    }
  }
  ret[len - 1] = size;
}

template <typename real_type, typename rng_state_type>
void multinomial(rng_state_type& rng_state,
                 int size,
                 const std::vector<real_type>& prob,
                 std::vector<real_type>& ret) {
  multinomial(rng_state, prob.data(), size, ret.data());
}

template <typename real_type, typename rng_state_type>
std::vector<real_type> multinomial(rng_state_type& rng_state,
                                   real_type size,
                                   const std::vector<real_type>& prob) {
  std::vector<real_type> ret(prob.size());
  multinomial(rng_state, prob, size, ret);
  return ret;
}

}
}

#endif
