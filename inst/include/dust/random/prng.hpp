#ifndef DUST_RANDOM_PRNG_HPP
#define DUST_RANDOM_PRNG_HPP

#include <vector>
#include <dust/random/generator.hpp>

namespace dust {
namespace random {

template <typename T>
class prng {
public:
  typedef T rng_state;
  typedef typename rng_state::int_type int_type;
  prng(const size_t n, const std::vector<int_type>& seed,
       const bool deterministic) {
    rng_state s;
    s.deterministic = deterministic;

    const size_t len = rng_state::size();
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

  size_t size() const {
    return state_.size();
  }

  void jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      dust::random::jump(state_[i]);
    }
  }

  void long_jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      dust::random::long_jump(state_[i]);
    }
  }

  rng_state& state(size_t i) {
    return state_[i];
  }

  std::vector<int_type> export_state() {
    const size_t n = rng_state::size();
    std::vector<int_type> state;
    state.resize(size() * n);
    for (size_t i = 0, k = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j, ++k) {
        state[k] = state_[i][j];
      }
    }
    return state;
  }

  void import_state(const std::vector<int_type>& state, const size_t len) {
    auto it = state.begin();
    const size_t n = rng_state::size();
    for (size_t i = 0; i < len; ++i) {
      for (size_t j = 0; j < n; ++j) {
        state_[i][j] = *it;
        ++it;
      }
    }
  }

  void import_state(const std::vector<int_type>& state) {
    import_state(state, size());
  }

  bool deterministic() const {
    return state_[0].deterministic;
  }

private:
  std::vector<rng_state> state_;
};

}
}

#endif
