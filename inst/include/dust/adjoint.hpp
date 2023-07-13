#ifndef DUST_ADJOINT_HPP
#define DUST_ADJOINT_HPP

#include <algorithm>
#include <map>

#include "dust/particle.hpp"

namespace dust {

template <typename real_type>
struct adjoint_data {
public:
  adjoint_data(size_t n_state, size_t n_adjoint, size_t n_time) :
    log_likelihood(0),
    gradient(n_adjoint - n_state),
    n_state_(n_state),
    n_adjoint_(n_adjoint),
    n_time_(n_time),
    state_(n_state * n_time),
    adjoint_curr_(n_adjoint),
    adjoint_next_(n_adjoint) {
  }

  void resize(size_t n_time) {
    n_time_ = n_time;
    if (n_time != n_time_) {
      state_.resize(n_state_ * n_time);
    }
  }

  real_type * state() {
    return state_.data();
  }

  real_type * adjoint_curr() {
    return adjoint_curr_.data();
  }

  real_type * adjoint_next() {
    return adjoint_next_.data();
  }

  void finish(real_type * adjoint) {
    if (adjoint == adjoint_next_.data()) {
      std::swap(adjoint_next_, adjoint_curr_);
    }
    std::copy(adjoint + n_state_, adjoint + n_adjoint_, gradient.begin());
  }

  real_type log_likelihood;
  std::vector<real_type> gradient;

private:
  size_t n_state_;
  size_t n_adjoint_;
  size_t n_time_;
  std::vector<real_type> state_;
  std::vector<real_type> adjoint_curr_;
  std::vector<real_type> adjoint_next_;
};


template <typename T>
using data_map_type = std::map<size_t, std::vector<typename T::data_type>>;

template <typename T>
adjoint_data<typename T::real_type>
adjoint(particle<T> particle,
        const data_map_type<T>& data) {
  using real_type = typename T::real_type;
  // Ideally typename particle<T>::time_type, but that does not work
  // generally, failing on mac and with nvcc. At the moment the time
  // type must be size_t anyway, so this is fine.
  using time_type = size_t;

  typename T::rng_state_type rng_state;
  rng_state.deterministic = true;
  for (size_t i = 0; i < T::rng_state_type::size(); ++i) {
    rng_state[i] = 0;
  }

  auto d_start = data.begin();
  auto d_end = data.end();

  T& model = particle.model();
  const time_type time_start = particle.time();
  const size_t n_state = model.size();
  const size_t n_adjoint = model.adjoint_size();

  if (time_start > d_start->first) {
    std::stringstream msg;
    msg << "Expected model start time (" << time_start <<
      ") to be at most the first data time (" << d_start->first << ")";
    throw std::runtime_error(msg.str());
  }

  // This is super ugly, just to get the last time, which we need in
  // order to get the the size of the space we need to hold the whole
  // simulation. Later we'll do this just once so it won't matter so
  // much.
  auto d_last = data.end();
  --d_last;
  const auto n_time = d_last->first - time_start + 1;

  // We will want one of these per thread, but we'll want to save some
  // back into the main object; we could use these from the parent
  // object perhaps.
  adjoint_data<real_type> result(n_state, n_adjoint, n_time);

  auto state_curr = result.state();
  auto state_next = state_curr + n_state;
  auto adjoint_curr = result.adjoint_curr();
  auto adjoint_next = result.adjoint_next();

  // We might not do this bit, generally, but instead take the initial
  // state from elsewhere (model.state() contains what we need here).
  const auto state_initial = model.initial(time_start, rng_state);
  std::copy_n(state_initial.begin(), n_state, state_curr);

  auto d = data.begin();

  // Forwards; compute the log likelihood from the initial conditions:
  time_type time = time_start;
  real_type ll = 0;
  while (d != d_end) {
    while (time < d->first) {
      model.update(time, state_curr, rng_state, state_next);
      state_curr = state_next;
      state_next += n_state;
      ++time;
    }
    ll += model.compare_data(state_curr, d->second[0], rng_state);
    ++d;
  }

  result.log_likelihood = ll;

  std::fill(adjoint_curr, adjoint_curr + n_adjoint, 0);
  d--;

  while (time > time_start) {
    if (time == d->first) {
      model.adjoint_compare_data(time, state_curr, d->second[0], adjoint_curr,
                                 adjoint_next);
      std::swap(adjoint_curr, adjoint_next);
    } else if (d != d_start && time < d->first) {
      --d;
    }
    state_curr -= n_state;
    --time;
    model.adjoint_update(time, state_curr, adjoint_curr, adjoint_next);
    std::swap(adjoint_curr, adjoint_next);
  }

  model.adjoint_initial(time, state_curr, adjoint_curr, adjoint_next);
  std::swap(adjoint_curr, adjoint_next);

  // Tidy up the object so that we agree on what the current set of
  // data is and copy the final gradient out.
  result.finish(adjoint_curr);

  return result;
}

}

#endif
