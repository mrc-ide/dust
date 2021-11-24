#ifndef DUST_RANDOM_MULTINOMIAL_HPP
#define DUST_RANDOM_MULTINOMIAL_HPP

#include <cmath>
#include <vector>

#include "dust/random/binomial.hpp"
#include "dust/random/generator.hpp"
#include "dust/random/numeric.hpp"

namespace dust {
namespace random {

/// Draw one sample from the multinomial distribution.
///
/// This is written assuming that `prob` and `ret` are arbitrary
/// containers; they could be pointers, vectors or anything else that
/// supports random access.  We could also do this with iterators but
/// they always feel a bit weird really.
///
/// Note that `prob` and `ret` will ordinarily be containers of
/// `real_type` but that might not be the case, in particular when
/// calling from R where we want calculations to happen in single
/// precision but the inputs (and destination) are double precision.
///
/// Later we will provide some helpers that will allow setting a
/// stride over these containers; this will help with use from odin
/// code.
///
/// @tparam real_type The underlying real number type, typically
/// `double` or `float`. A compile-time error will be thrown if you
/// attempt to use a non-floating point type (based on
/// `std::is_floating_point).
///
/// @tparam rng_state_type The random number state type
///
/// @tparam T,U The type of the containers for `prob` and `ret`. This
/// might be `double*` or `std::vector<double>` depending on use.
///
/// @param rng_state Reference to the random number state, will be
/// modified as a side-effect
///
/// @param size The number of trials (analagous to `size` in `binomial`)
///
/// @param prob The set of probabilities. In a binomial trial we only
/// provide a single probability of success but here there is a
/// vector/array of such probabilities. These need not sum to one as
/// they will be normalised, however they must all be non-negative and
/// at least one must be positive.
///
/// @param prob_len The number of probabilities (or outcomes)
///
/// @param ret Container for the return value
template <typename real_type, typename rng_state_type,
          typename T, typename U>
__host__ __device__
void multinomial(rng_state_type& rng_state, int size, const T& prob,
                 int prob_len, U& ret) {
  real_type p_tot = 0;
  for (int i = 0; i < prob_len; ++i) {
    if (prob[i] < 0) {
      dust::utils::fatal_error("Negative prob passed to multinomial");
    }
    p_tot += prob[i];
  }
  if (p_tot == 0) {
    dust::utils::fatal_error("No positive prob in call to multinomial");
  }

  for (int i = 0; i < prob_len - 1; ++i) {
    if (prob[i] > 0) {
      const real_type pi = utils::min(static_cast<real_type>(prob[i]) / p_tot,
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
  multinomial<real_type>(rng_state, size, prob, prob.size(), ret);
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
