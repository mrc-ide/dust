#ifndef DUST_INTERFACE_HPP
#define DUST_INTERFACE_HPP

#include <cstring>
#include <map>
#include <cpp11/doubles.hpp>
#include <cpp11/external_pointer.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/matrix.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/strings.hpp>

#include <dust/interface/random.hpp>
#include <dust/interface/helpers.hpp>
#include <dust/filter.hpp>

namespace dust {

template <typename T>
typename dust::pars_type<T> dust_pars(cpp11::list pars);

template <typename T>
typename T::data_type dust_data(cpp11::list data);

template <typename T>
cpp11::sexp dust_info(const dust::pars_type<T>& pars) {
  return R_NilValue;
}

namespace r {

template <typename T>
cpp11::list dust_alloc_cpu(cpp11::list r_pars, bool pars_multi, int step,
                           cpp11::sexp r_n_particles, int n_threads,
                           cpp11::sexp r_seed, bool deterministic) {
  typedef typename T::rng_state_type rng_state_type;
  dust::interface::validate_size(step, "step");
  dust::interface::validate_positive(n_threads, "n_threads");
  std::vector<typename rng_state_type::int_type> seed =
    dust::interface::as_rng_seed<rng_state_type>(r_seed);

  Dust<T> *d = nullptr;
  cpp11::sexp info;
  if (pars_multi) {
    dust::interface::check_pars_multi(r_pars);
    std::vector<dust::pars_type<T>> pars;
    cpp11::writable::list info_list = cpp11::writable::list(r_pars.size());
    for (int i = 0; i < r_pars.size(); ++i) {
      pars.push_back(dust_pars<T>(r_pars[i]));
      info_list[i] = dust_info<T>(pars[i]);
    }
    info = info_list;
    cpp11::sexp dim_pars = r_pars.attr("dim");
    std::vector<size_t> shape;
    if (dim_pars == R_NilValue) {
      shape.push_back(pars.size());
    } else {
      cpp11::integers dim_pars_int = cpp11::as_cpp<cpp11::integers>(dim_pars);
      for (int i = 0; i < dim_pars_int.size(); ++i) {
        shape.push_back(dim_pars_int[i]);
      }
    }
    size_t n_particles = 0;
    if (r_n_particles != R_NilValue) {
      n_particles = cpp11::as_cpp<int>(r_n_particles);
      dust::interface::validate_size(n_particles, "n_particles");
    }
    d = new Dust<T>(pars, step, n_particles, n_threads, seed, deterministic,
                    shape);
  } else {
    size_t n_particles = cpp11::as_cpp<int>(r_n_particles);
    dust::interface::validate_positive(n_particles, "n_particles");
    dust::pars_type<T> pars = dust_pars<T>(r_pars);
    d = new Dust<T>(pars, step, n_particles, n_threads, seed, deterministic);
    info = dust_info<T>(pars);
  }
  cpp11::external_pointer<Dust<T>> ptr(d, true, false);

  cpp11::writable::integers r_shape =
    dust::interface::vector_size_to_int(d->shape());

  return cpp11::writable::list({ptr, info, r_shape});
}

template <typename T>
typename std::enable_if<!dust::has_gpu_support<T>::value, cpp11::list>::type
dust_alloc_device(cpp11::list r_pars, bool pars_multi, int step,
                  cpp11::sexp r_n_particles, int n_threads,
                  cpp11::sexp r_seed, bool deterministic,
                  cpp11::sexp device_info) {
  cpp11::stop("This model does not have GPU support");
  return R_NilValue; // #nocov - can't get here
}

template <typename T>
typename std::enable_if<dust::has_gpu_support<T>::value, cpp11::list>::type
dust_alloc_device(cpp11::list r_pars, bool pars_multi, int step,
                  cpp11::sexp r_n_particles, int n_threads,
                  cpp11::sexp r_seed, bool deterministic,
                  cpp11::sexp device_info) {
  cpp11::stop("Still need to implement this");
  return R_NilValue;
}

template <typename T>
cpp11::list dust_alloc(cpp11::list r_pars, bool pars_multi, int step,
                       cpp11::sexp r_n_particles, int n_threads,
                       cpp11::sexp r_seed, bool deterministic,
                       cpp11::sexp device_config) {
  if (device_config == R_NilValue) {
    return dust_alloc_cpu<T>(r_pars, pars_multi, step, r_n_particles,
                             n_threads, r_seed, deterministic);
  } else {
    return dust_alloc_device<T>(r_pars, pars_multi, step, r_n_particles,
                                n_threads, r_seed, deterministic,
                                device_config);
  }
}

template <typename T>
void dust_set_index(SEXP ptr, cpp11::sexp r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index =
    dust::interface::r_index_to_index(r_index, index_max);
  obj->set_index(index);
}

template <typename T>
cpp11::sexp dust_update_state_set_pars(Dust<T> *obj, cpp11::list r_pars,
                                       bool set_initial_state) {
  cpp11::sexp ret = R_NilValue;
  if (obj->n_pars() == 0) {
    dust::pars_type<T> pars = dust_pars<T>(r_pars);
    obj->set_pars(pars, set_initial_state);
    ret = dust_info<T>(pars);
  } else {
    dust::interface::check_pars_multi(r_pars, obj->shape(),
                                      obj->pars_are_shared());
    std::vector<dust::pars_type<T>> pars;
    pars.reserve(obj->n_pars());
    cpp11::writable::list info_list =
      cpp11::writable::list(r_pars.size());
    for (int i = 0; i < r_pars.size(); ++i) {
      pars.push_back(dust_pars<T>(r_pars[i]));
      info_list[i] = dust_info<T>(pars[i]);
    }
    obj->set_pars(pars, set_initial_state);
    ret = info_list;
  }
  return ret;
}

// There are many components of state (not including rng state which
// we treat separately), we set components always in the order (1)
// pars, (2) state, (3) step
template <typename T>
cpp11::sexp dust_update_state_set(Dust<T> *obj, SEXP r_pars,
                                  const std::vector<typename T::real_type>& state,
                                  const std::vector<size_t>& step,
                                  bool set_initial_state) {
  cpp11::sexp ret = R_NilValue;
  if (r_pars != R_NilValue) {
    ret = dust_update_state_set_pars(obj, cpp11::as_cpp<cpp11::list>(r_pars),
                               set_initial_state);
  }

  if (state.size() > 0) {
    obj->set_state(state);
  }

  if (step.size() == 1) {
    obj->set_step(step[0]);
  } else if (step.size() > 1) {
    obj->set_step(step);
  }

  // If we set both initial conditions and step then we're safe to
  // continue here.
  if (state.size() > 0 && step.size() > 0) {
    obj->reset_errors();
  }

  return ret;
}

template <typename T>
SEXP dust_update_state(SEXP ptr, SEXP r_pars, SEXP r_state, SEXP r_step,
                       SEXP r_set_initial_state) {
  typedef typename T::real_type real_type;
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();

  bool set_initial_state = false;
  if (r_set_initial_state == R_NilValue) {
    set_initial_state = r_state == R_NilValue &&
      r_pars != R_NilValue && r_step != R_NilValue;
  } else {
    set_initial_state = cpp11::as_cpp<bool>(r_set_initial_state);
  }

  if (set_initial_state && r_pars == R_NilValue) {
    cpp11::stop("Can't use 'set_initial_state' without providing 'pars'");
  }
  if (set_initial_state && r_state != R_NilValue) {
    cpp11::stop("Can't use both 'set_initial_state' and provide 'state'");
  }

  // Do the validation on both arguments first so that we leave this
  // function having dealt with both or neither (i.e., do not fail on
  // step after succeeding on state).

  std::vector<size_t> step;
  std::vector<real_type> state;

  if (r_step != R_NilValue) {
    // TODO: what about the length and dimensions here? What is the
    // best thing to take for those? Possibly require an array to
    // disambiguate?
    step = dust::interface::validate_size(r_step, "step");
    if (!(step.size() == 1 || step.size() == obj->n_particles())) {
      cpp11::stop("Expected 'step' to be scalar or length %d",
                  obj->n_particles());
    }
  }

  if (r_state != R_NilValue) {
    state = dust::interface::check_state<real_type>(r_state,
                                                    obj->n_state_full(),
                                                    obj->shape(),
                                                    obj->pars_are_shared());
  }

  return dust_update_state_set(obj, r_pars, state, step, set_initial_state);
}

template <typename T>
cpp11::sexp dust_run(SEXP ptr, int step_end) {
  dust::interface::validate_size(step_end, "step_end");
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->check_errors();
  obj->run(step_end);

  // TODO: the allocation should come from the dust object, *or* we
  // should be able to do this with a pointer to the C array. The
  // current version helps noone.
  std::vector<typename T::real_type> dat(obj->n_state() * obj->n_particles());
  if (obj->n_state() > 0) {
    obj->state(dat);
  }

  return dust::interface::state_array(dat, obj->n_state(), obj->shape());
}

template <typename T>
cpp11::sexp dust_simulate(SEXP ptr, cpp11::sexp r_step_end) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->check_errors();
  const std::vector<size_t> step_end =
    dust::interface::validate_size(r_step_end, "step_end");
  const size_t n_time = step_end.size();
  if (n_time == 0) {
    cpp11::stop("'step_end' must have at least one element");
  }
  if (step_end[0] < obj->step()) {
    cpp11::stop("'step_end[1]' must be at least %d", obj->step());
  }
  for (size_t i = 1; i < n_time; ++i) {
    if (step_end[i] < step_end[i - 1]) {
      cpp11::stop("'step_end' must be non-decreasing (error on element %d)",
                  i + 1);
    }
  }

  std::vector<typename T::real_type> dat = obj->simulate(step_end);

  return dust::interface::state_array(dat, obj->n_state(), obj->shape(),
                                      n_time);
}

template <typename T>
SEXP dust_state_full(Dust<T> *obj) {
  const size_t n_state_full = obj->n_state_full();
  const size_t len = n_state_full * obj->n_particles();

  std::vector<typename T::real_type> dat(len);
  obj->state_full(dat);

  return dust::interface::state_array(dat, n_state_full, obj->shape());
}

template <typename T>
SEXP dust_state_select(Dust<T> *obj, cpp11::sexp r_index) {
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index =
    dust::interface::r_index_to_index(r_index, index_max);
  const size_t n_state = static_cast<size_t>(index.size());
  const size_t len = n_state * obj->n_particles();

  std::vector<typename T::real_type> dat(len);
  obj->state(index, dat);

  return dust::interface::state_array(dat, n_state, obj->shape());
}

template <typename T>
SEXP dust_state(SEXP ptr, SEXP r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  if (r_index == R_NilValue) {
    return dust_state_full(obj);
  } else {
    return dust_state_select(obj, r_index);
  }
}

template <typename T>
size_t dust_step(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->check_errors();
  return obj->step();
}

template <typename T>
void dust_reorder(SEXP ptr, cpp11::sexp r_index) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  std::vector<size_t> index =
    dust::interface::check_reorder_index(r_index, obj->shape());
  obj->reorder(index);
}

template <typename T>
SEXP dust_resample(SEXP ptr, cpp11::doubles r_weights) {
  typedef typename T::real_type real_type;

  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  size_t n_particles = obj->n_particles();
  size_t n_pars = obj->n_pars_effective();
  size_t n_particles_each = n_particles / n_pars;

  std::vector<real_type> weights =
    dust::interface::check_resample_weights<real_type>(r_weights, obj->shape());
  std::vector<size_t> idx = obj->resample(weights);

  cpp11::writable::integers ret(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    ret[i] = idx[i] % n_particles_each + 1;
  }

  // Same shape as on exit; we rescale the index so that it is
  // equivalent to the order that reorder would accept
  ret.attr("dim") = r_weights.attr("dim");
  return ret;
}

template <typename T>
SEXP dust_rng_state(SEXP ptr, bool first_only, bool last_only) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  typedef typename T::rng_state_type rng_state_type;
  auto state = obj->rng_state();
  if (first_only && last_only) {
    cpp11::stop("Only one of 'first_only' or 'last_only' may be TRUE");
  }
  size_t n = (first_only || last_only) ?
    rng_state_type::size() : state.size();
  size_t rng_offset = last_only ? obj->n_particles() * n : 0;
  size_t len = sizeof(typename rng_state_type::int_type) * n;
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data() + rng_offset, len);
  return ret;
}

template <typename T>
void dust_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  typedef typename T::rng_state_type::int_type int_type;
  auto prev_state = obj->rng_state();
  size_t len = prev_state.size() * sizeof(int_type);
  if ((size_t)rng_state.size() != len) {
    cpp11::stop("'rng_state' must be a raw vector of length %d (but was %d)",
                len, rng_state.size());
  }
  std::vector<int_type> pars(prev_state.size());
  std::memcpy(pars.data(), RAW(rng_state), len);
  obj->set_rng_state(pars);
}

template <typename T>
void dust_set_n_threads(SEXP ptr, int n_threads) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  dust::interface::validate_positive(n_threads, "n_threads");
  obj->set_n_threads(n_threads);
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
void dust_set_data(SEXP ptr, cpp11::list r_data) {
  typedef typename T::data_type data_type;
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  const size_t n_pars = obj->n_pars_effective();

  const size_t len = r_data.size();
  std::map<size_t, std::vector<data_type>> data;

  for (size_t i = 0; i < len; ++i) {
    cpp11::list el = r_data[i];
    if (el.size() != static_cast<int>(n_pars) + 1) {
      cpp11::stop("Expected a list of length %d for element %d of 'data'",
                  n_pars + 1, i + 1);
    }
    const size_t step_i = cpp11::as_cpp<int>(el[0]);
    std::vector<data_type> data_i;
    data_i.reserve(n_pars);
    for (size_t j = 0; j < n_pars; ++j) {
      data_i.push_back(dust_data<T>(cpp11::as_cpp<cpp11::list>(el[j + 1])));
    }
    data[step_i] = data_i;
  }
  obj->set_data(data);
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
cpp11::sexp dust_compare_data(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->check_errors();

  std::vector<typename T::real_type> ret = obj->compare_data();

  if (ret.size() == 0) {
    return R_NilValue;
  }
  cpp11::writable::doubles ret_r(ret.size());
  std::copy(ret.begin(), ret.end(), REAL(ret_r));
  if (obj->shape().size() > 1) {
    ret_r.attr("dim") =
      cpp11::writable::integers(obj->shape().begin(), obj->shape().end());
  }

  return ret_r;
}

template <typename filter_state, typename T>
cpp11::sexp save_trajectories(const filter_state& trajectories,
                              const Dust<T> *obj) {
  cpp11::writable::doubles trajectories_data(trajectories.size());
  trajectories.history(REAL(trajectories_data));
  trajectories_data.attr("dim") =
    dust::interface::state_array_dim(obj->n_state(), obj->shape(),
                                      obj->n_data() + 1);
  cpp11::sexp r_trajectories = trajectories_data;
  return(r_trajectories);
}

template <typename filter_state, typename T>
cpp11::sexp save_snapshots(const filter_state& snapshots, const Dust<T> *obj,
                           const std::vector<size_t>& step_snapshot) {
  cpp11::writable::doubles snapshots_data(snapshots.size());
  snapshots.history(REAL(snapshots_data));
  snapshots_data.attr("dim") =
    dust::interface::state_array_dim(obj->n_state_full(), obj->shape(),
                                     step_snapshot.size());
  cpp11::sexp r_snapshots = snapshots_data;
  return(r_snapshots);
}

// TODO: this should be rewritten to return a list of 3 cpp1::sexp not
// passing two as references.
template <typename T, typename state_type>
cpp11::writable::doubles run_filter(Dust<T> * obj,
                                    cpp11::sexp& r_trajectories,
                                    cpp11::sexp& r_snapshots,
                                    std::vector<size_t>& step_snapshot,
                                    bool save_trajectories) {
  state_type filter_state;
  cpp11::writable::doubles log_likelihood =
    dust::filter::filter(obj, filter_state,
                         save_trajectories, step_snapshot);
  if (save_trajectories) {
    r_trajectories = dust::r::save_trajectories(filter_state.trajectories, obj);
  }
  if (!step_snapshot.empty()) {
    r_snapshots = dust::r::save_snapshots(filter_state.snapshots, obj, step_snapshot);
  }
  return log_likelihood;
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
cpp11::sexp dust_filter(SEXP ptr, bool save_trajectories,
                        cpp11::sexp r_step_snapshot) {
  typedef typename T::real_type real_type;
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  obj->check_errors();

  if (obj->data().empty()) {
    cpp11::stop("Data has not been set for this object");
  }

  std::vector<size_t> step_snapshot =
      dust::interface::check_step_snapshot(r_step_snapshot, obj->data());

  cpp11::sexp r_trajectories, r_snapshots;
  cpp11::writable::doubles log_likelihood =
    run_filter<T, dust::filter::filter_state_host<real_type>>
    (obj, r_trajectories, r_snapshots, step_snapshot, save_trajectories);

  using namespace cpp11::literals;
  return cpp11::writable::list({"log_likelihood"_nm = log_likelihood,
                                "trajectories"_nm = r_trajectories,
                                "snapshots"_nm = r_snapshots});
}

// Based on the value of the data_type in the underlying model class we
// might use these functions for set_data and compare_data which give
// reasonable errors back to R, as we can't use the full versions
// above.
inline void disable_method(const char * name) {
  cpp11::stop("The '%s' method is not supported for this class", name);
}

template <typename T, typename std::enable_if<std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
void dust_set_data(SEXP ptr, cpp11::list r_data) {
  disable_method("set_data");
}

template <typename T, typename std::enable_if<std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
cpp11::sexp dust_compare_data(SEXP ptr) {
  disable_method("compare_data");
  return R_NilValue; // #nocov never gets here
}

template <typename T, typename std::enable_if<std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
cpp11::sexp dust_filter(SEXP ptr, bool save_trajectories,
                        cpp11::sexp step_snapshot) {
  disable_method("filter");
  return R_NilValue; // #nocov never gets here
}

template <typename T>
cpp11::sexp dust_capabilities() {
  using namespace cpp11::literals;
#ifdef _OPENMP
  bool openmp = true;
#else
  bool openmp = false;
#endif
#ifdef __NVCC__
  bool cuda = true;
#else
  bool cuda = false;
#endif
  bool compare = !std::is_same<dust::no_data, typename T::data_type>::value;
  return cpp11::writable::list({"openmp"_nm = openmp,
                                "compare"_nm = compare,
                                "cuda"_nm = cuda});
}

template <typename T>
int dust_n_state(SEXP ptr) {
  Dust<T> *obj = cpp11::as_cpp<cpp11::external_pointer<Dust<T>>>(ptr).get();
  return obj->n_state_full();
}

}
}

#endif
