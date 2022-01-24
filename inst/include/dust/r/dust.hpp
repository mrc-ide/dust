#ifndef DUST_R_DUST_HPP
#define DUST_R_DUST_HPP

#include <cstring>
#include <map>

#include <cpp11/doubles.hpp>
#include <cpp11/external_pointer.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/matrix.hpp>
#include <cpp11/raws.hpp>
#include <cpp11/strings.hpp>

#include "dust/gpu/dust_gpu.hpp"
#include "dust/gpu/filter.hpp"
#include "dust/dust_cpu.hpp"
#include "dust/filter.hpp"

#include "dust/r/gpu.hpp"
#include "dust/r/helpers.hpp"
#include "dust/r/random.hpp"

namespace dust {
namespace r {

template <typename T>
cpp11::list dust_cpu_alloc(cpp11::list r_pars, bool pars_multi, int step,
                           cpp11::sexp r_n_particles, int n_threads,
                           cpp11::sexp r_seed, bool deterministic,
                           cpp11::sexp r_gpu_config) {
  dust_cpu<T> *d = nullptr;
  cpp11::sexp info;
  if (pars_multi) {
    auto inputs =
      dust::r::process_inputs_multi<T>(r_pars, step, r_n_particles,
                                       n_threads, r_seed);
    info = inputs.info;
    d = new dust_cpu<T>(inputs.pars, inputs.step, inputs.n_particles,
                        inputs.n_threads, inputs.seed, deterministic,
                        inputs.shape);
  } else {
    auto inputs =
      dust::r::process_inputs_single<T>(r_pars, step, r_n_particles,
                                        n_threads, r_seed);
    info = inputs.info;
    d = new dust_cpu<T>(inputs.pars[0], inputs.step, inputs.n_particles,
                        inputs.n_threads, inputs.seed, deterministic);
  }
  cpp11::external_pointer<dust_cpu<T>> ptr(d, true, false);

  cpp11::writable::integers r_shape =
    dust::r::vector_size_to_int(d->shape());

  return cpp11::writable::list({ptr, info, r_shape, R_NilValue});
}

template <typename T>
cpp11::list dust_gpu_alloc(cpp11::list r_pars, bool pars_multi, int step,
                           cpp11::sexp r_n_particles, int n_threads,
                           cpp11::sexp r_seed, bool deterministic,
                           cpp11::sexp r_gpu_config) {
  const dust::gpu::gpu_config gpu_config =
    dust::gpu::r::gpu_config(r_gpu_config);
  if (deterministic) {
    cpp11::stop("Deterministic models not supported on gpu");
  }

  dust_gpu<T> *d = nullptr;
  cpp11::sexp info;
  if (pars_multi) {
    auto inputs =
      dust::r::process_inputs_multi<T>(r_pars, step, r_n_particles,
                                       n_threads, r_seed);
    info = inputs.info;
    d = new dust_gpu<T>(inputs.pars, inputs.step, inputs.n_particles,
                        inputs.n_threads, inputs.seed,
                        inputs.shape, gpu_config);
  } else {
    auto inputs =
      dust::r::process_inputs_single<T>(r_pars, step, r_n_particles,
                                        n_threads, r_seed);
    info = inputs.info;
    d = new dust_gpu<T>(inputs.pars[0], inputs.step, inputs.n_particles,
                        inputs.n_threads, inputs.seed, gpu_config);
  }
  cpp11::external_pointer<dust_gpu<T>> ptr(d, true, false);

  cpp11::writable::integers r_shape =
    dust::r::vector_size_to_int(d->shape());

  cpp11::sexp ret_r_gpu_config =
    dust::gpu::r::gpu_config_as_sexp(gpu_config);

  return cpp11::writable::list({ptr, info, r_shape, ret_r_gpu_config});
}

template <typename T>
void dust_set_index(SEXP ptr, cpp11::sexp r_index) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index =
    dust::r::r_index_to_index(r_index, index_max);
  obj->set_index(index);
}

template <typename T>
cpp11::sexp dust_update_state_set_pars(T *obj, cpp11::list r_pars,
                                       bool set_initial_state) {
  using model_type = typename T::model_type;
  cpp11::sexp ret = R_NilValue;
  if (obj->n_pars() == 0) {
    dust::pars_type<model_type> pars = dust_pars<model_type>(r_pars);
    obj->set_pars(pars, set_initial_state);
    ret = dust_info<model_type>(pars);
  } else {
    dust::r::check_pars_multi(r_pars, obj->shape(),
                              obj->pars_are_shared());
    std::vector<dust::pars_type<model_type>> pars;
    pars.reserve(obj->n_pars());
    cpp11::writable::list info_list =
      cpp11::writable::list(r_pars.size());
    for (int i = 0; i < r_pars.size(); ++i) {
      pars.push_back(dust_pars<model_type>(r_pars[i]));
      info_list[i] = dust_info<model_type>(pars[i]);
    }
    obj->set_pars(pars, set_initial_state);
    ret = info_list;
  }
  return ret;
}

// There are many components of state (not including rng state which
// we treat separately), we set components always in the order (1)
// step, (2) pars, (3) state
template <typename T, typename real_type>
cpp11::sexp dust_update_state_set(T *obj, SEXP r_pars,
                                  const std::vector<real_type>& state,
                                  const std::vector<size_t>& step,
                                  bool set_initial_state) {
  cpp11::sexp ret = R_NilValue;
  const size_t step_prev = obj->step();

  if (step.size() == 1) { // TODO: can handle this via a bool and int, tidier
    obj->set_step(step[0]);
  }

  if (r_pars != R_NilValue) {
    // NOTE: if initial state is step dependent, then this picks up
    // the step set above.
    try {
      ret = dust_update_state_set_pars(obj, cpp11::as_cpp<cpp11::list>(r_pars),
                                       set_initial_state);
    } catch (const std::invalid_argument& e) {
      obj->set_step(step_prev);
      throw e;
    }
  }

  if (state.size() > 0) { // && !set_initial_state, though that is implied
    obj->set_state(state);
  }

  // If we set both initial conditions and step then we're safe to
  // continue here.
  if ((set_initial_state || state.size() > 0) && step.size() > 0) {
    obj->reset_errors();
  }

  return ret;
}

template <typename T>
SEXP dust_update_state(SEXP ptr, SEXP r_pars, SEXP r_state, SEXP r_step,
                       SEXP r_set_initial_state) {
  using real_type = typename T::real_type;
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();

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
    // TODO: simplify this, if possible
    step = dust::r::validate_size(r_step, "step");
    const size_t len = step.size();
    if (len != 1) {
      cpp11::stop("Expected 'step' to be scalar");
    }
  }

  if (r_state != R_NilValue) {
    state = dust::r::check_state<real_type>(r_state,
                                            obj->n_state_full(),
                                            obj->shape(),
                                            obj->pars_are_shared());
  }

  return dust_update_state_set(obj, r_pars, state, step, set_initial_state);
}

template <typename T>
cpp11::sexp dust_run(SEXP ptr, int step_end) {
  dust::r::validate_size(step_end, "step_end");
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  obj->check_errors();
  obj->run(step_end);

  // TODO: the allocation should come from the dust object, *or* we
  // should be able to do this with a pointer to the C array. The
  // current version helps noone.
  std::vector<typename T::real_type> dat(obj->n_state() * obj->n_particles());
  if (obj->n_state() > 0) {
    obj->state(dat);
  }

  return dust::r::state_array(dat, obj->n_state(), obj->shape());
}

template <typename T>
cpp11::sexp dust_simulate(SEXP ptr, cpp11::sexp r_step_end) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  obj->check_errors();
  const std::vector<size_t> step_end =
    dust::r::validate_size(r_step_end, "step_end");
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

  return dust::r::state_array(dat, obj->n_state(), obj->shape(),
                              n_time);
}

template <typename T>
SEXP dust_state_full(T *obj) {
  const size_t n_state_full = obj->n_state_full();
  const size_t len = n_state_full * obj->n_particles();

  std::vector<typename T::real_type> dat(len);
  obj->state_full(dat);

  return dust::r::state_array(dat, n_state_full, obj->shape());
}

template <typename T>
SEXP dust_state_select(T *obj, cpp11::sexp r_index) {
  const size_t index_max = obj->n_state_full();
  const std::vector<size_t> index =
    dust::r::r_index_to_index(r_index, index_max);
  const size_t n_state = static_cast<size_t>(index.size());
  const size_t len = n_state * obj->n_particles();

  std::vector<typename T::real_type> dat(len);
  obj->state(index, dat);

  return dust::r::state_array(dat, n_state, obj->shape());
}

template <typename T>
SEXP dust_state(SEXP ptr, SEXP r_index) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  if (r_index == R_NilValue) {
    return dust_state_full(obj);
  } else {
    return dust_state_select(obj, r_index);
  }
}

template <typename T>
size_t dust_step(SEXP ptr) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  obj->check_errors();
  return obj->step();
}

template <typename T>
void dust_reorder(SEXP ptr, cpp11::sexp r_index) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  std::vector<size_t> index =
    dust::r::check_reorder_index(r_index, obj->shape());
  obj->reorder(index);
}

template <typename T>
SEXP dust_resample(SEXP ptr, cpp11::doubles r_weights) {
  using real_type = typename T::real_type;

  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  size_t n_particles = obj->n_particles();
  size_t n_pars = obj->n_pars_effective();
  size_t n_particles_each = n_particles / n_pars;

  std::vector<real_type> weights =
    dust::r::check_resample_weights<real_type>(r_weights, obj->shape());
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
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  using rng_state_type = typename T::rng_state_type;
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
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  using int_type = typename T::rng_state_type::int_type;
  auto prev_state = obj->rng_state();
  size_t len = prev_state.size() * sizeof(int_type);
  if ((size_t)rng_state.size() != len) {
    cpp11::stop("'rng_state' must be a raw vector of length %d (but was %d)",
                len, rng_state.size());
  }
  std::vector<int_type> state(prev_state.size());
  std::memcpy(state.data(), RAW(rng_state), len);
  obj->set_rng_state(state);
}

template <typename T>
void dust_set_n_threads(SEXP ptr, int n_threads) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  dust::r::validate_positive(n_threads, "n_threads");
  obj->set_n_threads(n_threads);
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
void dust_set_data(SEXP ptr, cpp11::list r_data) {
  using model_type = typename T::model_type;
  using data_type = typename T::data_type;
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
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
      // TODO: no reason why dust_data<T> could not work here, really?
      data_i.push_back(dust_data<model_type>(cpp11::as_cpp<cpp11::list>(el[j + 1])));
    }
    data[step_i] = data_i;
  }
  obj->set_data(data);
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
cpp11::sexp dust_compare_data(SEXP ptr) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
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
                              const T *obj) {
  cpp11::writable::doubles trajectories_data(trajectories.size());
  trajectories.history(REAL(trajectories_data));
  trajectories_data.attr("dim") =
    dust::r::state_array_dim(obj->n_state(), obj->shape(),
                             obj->n_data() + 1);
  cpp11::sexp r_trajectories = trajectories_data;
  return(r_trajectories);
}

template <typename filter_state, typename T>
cpp11::sexp save_snapshots(const filter_state& snapshots, const T *obj,
                           const std::vector<size_t>& step_snapshot) {
  cpp11::writable::doubles snapshots_data(snapshots.size());
  snapshots.history(REAL(snapshots_data));
  snapshots_data.attr("dim") =
    dust::r::state_array_dim(obj->n_state_full(), obj->shape(),
                             step_snapshot.size());
  cpp11::sexp r_snapshots = snapshots_data;
  return(r_snapshots);
}

template <typename T>
cpp11::sexp run_filter(T * obj, size_t step,
                       std::vector<size_t>& step_snapshot,
                       bool save_trajectories) {
  typename T::filter_state_type filter_state;
  cpp11::writable::doubles log_likelihood =
    dust::filter::filter(obj, step, filter_state, save_trajectories,
                         step_snapshot);
  cpp11::sexp r_trajectories, r_snapshots;
  if (save_trajectories) {
    r_trajectories = dust::r::save_trajectories(filter_state.trajectories, obj);
  }
  if (!step_snapshot.empty()) {
    r_snapshots = dust::r::save_snapshots(filter_state.snapshots, obj,
                                          step_snapshot);
  }

  using namespace cpp11::literals;
  return cpp11::writable::list({"log_likelihood"_nm = log_likelihood,
                                "trajectories"_nm = r_trajectories,
                                "snapshots"_nm = r_snapshots});
}

template <typename T, typename std::enable_if<!std::is_same<dust::no_data, typename T::data_type>::value, int>::type = 0>
cpp11::sexp dust_filter(SEXP ptr, SEXP r_step, bool save_trajectories,
                        cpp11::sexp r_step_snapshot,
                        cpp11::sexp min_log_likelihood) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  obj->check_errors();

  if (obj->n_data() == 0) {
    cpp11::stop("Data has not been set for this object");
  }

  size_t step = std::numeric_limits<size_t>::max();
  if (r_step != R_NilValue) {
    step = cpp11::as_cpp<int>(r_step);
    dust::r::validate_size(step, "step");
    if (step <= obj->step()) {
      cpp11::stop("'step' must be larger then curent step (%d; was given %d)",
                  obj->step(), step);
    }
    if (obj->data().find(step) == obj->data().end()) {
      cpp11::stop("'step' was not found in data (was given %d)", step);
    }
  }

  std::vector<size_t> step_snapshot =
    dust::r::check_step_snapshot(r_step_snapshot, obj->data());

  if (min_log_likelihood != R_NilValue) {
    cpp11::stop("min_log_likelihood not yet supported");
  }

  return run_filter<T>(obj, step, step_snapshot, save_trajectories);
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
cpp11::sexp dust_filter(SEXP ptr, SEXP step, bool save_trajectories,
                        cpp11::sexp step_snapshot,
                        cpp11::sexp min_log_likelihood) {
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
  bool gpu = true;
#else
  bool gpu = false;
#endif
  bool compare = !std::is_same<dust::no_data, typename T::data_type>::value;
  using real_type = typename T::real_type;
  auto real_size = sizeof(real_type);
  auto rng_algorithm =
    dust::random::r::algorithm_name<typename T::rng_state_type>();
  return cpp11::writable::list({"openmp"_nm = openmp,
                                "compare"_nm = compare,
                                "gpu"_nm = gpu,
                                "rng_algorithm"_nm = rng_algorithm,
                                "real_size"_nm = real_size * CHAR_BIT});
}

template <typename T>
int dust_n_state(SEXP ptr) {
  T *obj = cpp11::as_cpp<cpp11::external_pointer<T>>(ptr).get();
  return obj->n_state_full();
}

}
}

#endif
