#ifndef DUST_RANDOM_MULTINOMIAL_HPP
#define DUST_RANDOM_MULTINOMIAL_HPP

#include <cmath>
#include <dust/random/numeric.hpp>

namespace dust {
namespace random {

// This is written assuming tthat 'prob' and 'ret' are arbitrary
// containers; they could be pointers, vectors or anything else that
// supports random access.  We cold also do this with iterators but
// they always feel a bit weird really.
//
// Note that 'prob' and 'ret' (T) will ordinarily be real_type but
// that might not be the case, in particular when calling from R where
// we want calculations to happen on floats but the inputs (and
// destination) are double precision.
template <typename real_type, typename rng_state_type,
          typename T, typename U>
void multinomial(rng_state_type& rng_state, int size, const T& prob,
                 int prob_len, U& ret) {
  real_type p_tot = 0;
  for (int i = 0; i < prob_len; ++i) {
    p_tot += prob[i];
  }

  for (int i = 0; i < prob_len - 1; ++i) {
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
  ret[prob_len - 1] = size;
}

// These ones are designed for us within standalone programs and won't
// actually be tested by default which is not great.
template <typename real_type, typename rng_state_type>
void multinomial(rng_state_type& rng_state,
                 int size,
                 const std::vector<real_type>& prob,
                 std::vector<real_type>& ret) {
  multinomial<real_type>(rng_state, size, prob.data(), prob.size(), ret.data());
}

template <typename real_type, typename rng_state_type>
std::vector<real_type> multinomial(rng_state_type& rng_state,
                                   real_type size,
                                   const std::vector<real_type>& prob) {
  std::vector<real_type> ret(prob.size());
  multinomial(rng_state, size, prob, ret);
  return ret;
}

}
}

#endif
