#ifndef DUST_RANDOM_PRNG_HPP
#define DUST_RANDOM_PRNG_HPP

#include <algorithm>
#include <vector>

#include "dust/random/generator.hpp"

namespace dust {
namespace random {

/// Container class for parallel random number streams. This class
/// does not do any actual running of random numbers (and nothing in
/// parallel) but acts to hold together the state and ease some
/// bookkeeping.
///
/// @tparam T Random number state type to use
template <typename T>
class prng {
public:
  /// The random number state type
  using rng_state = T;
  /// The underlying integer type used by `rng_state`
  using int_type = typename rng_state::int_type;

  /// Construct a new `prng` object from a single integer seed
  /// @param n The number of streams
  /// @param seed An integer to use as a seed
  /// @param deterministic Selects use of the "deterministic" generator
  prng(const size_t n, const int seed, const bool deterministic = false) :
    prng(n, seed_data<T>(seed), deterministic) {
  }

  /// Construct a new `prng` object from a vector of seed data. We
  /// will consume as many items of `seed` as possible, then start
  /// jumping
  ///
  /// @param seed A vector of integers to seed the generator with
  prng(const size_t n, const std::vector<int_type>& seed,
       const bool deterministic = false) {
    rng_state s;
    s.deterministic = deterministic;

    constexpr size_t len = rng_state::size();
    auto n_seed = seed.size() / len;
    for (size_t i = 0; i < n; ++i) {
      if (i < n_seed) {
        std::copy_n(seed.begin() + i * len, len, std::begin(s.state));
      } else {
        dust::random::jump(s);
      }
      state_.push_back(s);
    }
  }

  /// The number of streams within the object
  size_t size() const {
    return state_.size();
  }

  /// Jump all generators forward
  void jump() {
    // TODO: I think this should be removed
    for (size_t i = 0; i < state_.size(); ++i) {
      dust::random::jump(state_[i]);
    }
  }

  /// Take a long jump for every generator
  void long_jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      dust::random::long_jump(state_[i]);
    }
  }

  /// Return the `i`th state, as an `rng_state` reference. This is the
  /// workhorse method and the main one likely to be used once the
  /// object is constructed.
  ///
  /// @param i The index of the stream (0, 1, ..., `size() - 1`)
  rng_state& state(size_t i) {
    return state_[i];
  }

  /// Convert the random number state of all generators into a single
  /// vector. This can be used to save the state to restore using
  /// `import_state()`
  std::vector<int_type> export_state() {
    constexpr size_t n = rng_state::size();
    std::vector<int_type> state;
    state.resize(size() * n);
    for (size_t i = 0, k = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j, ++k) {
        state[k] = state_[i][j];
      }
    }
    return state;
  }

  /// Import a vector of random number state, previously saved by
  /// `export_state()`
  ///
  /// @param state A vector of state
  void import_state(const std::vector<int_type>& state) {
    // TODO: check size
    auto it = state.begin();
    constexpr size_t n = rng_state::size();
    for (size_t i = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j) {
        state_[i][j] = *it;
        ++it;
      }
    }
  }

  /// Indicates if the generators are deterministic
  bool deterministic() const {
    return state_[0].deterministic;
  }

private:
  std::vector<rng_state> state_;
};

}
}

#endif
