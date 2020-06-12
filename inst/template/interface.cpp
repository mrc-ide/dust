#include <dust/dust.hpp>
#include <dust/util.hpp>

{{model}}

extern "C" void {{name}}_finalise(SEXP ptr) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, false);
  if (obj) {
    delete obj;
  }
  if (ptr) {
    R_ClearExternalPtr(ptr);
  }
}

extern "C" SEXP {{name}}_alloc(SEXP data, SEXP r_step,
                               SEXP r_n_particles, SEXP r_n_threads,
                               SEXP r_n_generators, SEXP r_seed) {
  size_t step = dust::util::as_size(r_step, "step");
  size_t n_particles = dust::util::as_size(r_n_particles, "n_particles");
  size_t n_threads = dust::util::as_size(r_n_threads, "n_threads");
  size_t n_generators = dust::util::as_size(r_n_generators, "n_generators");
  size_t seed = dust::util::as_size(r_seed, "seed");
  dust::util::validate_n(n_generators, n_threads);

  std::vector<size_t> index_y = {0};

  dust::Dust<{{type}}> *d =
    new dust::Dust<{{type}}>(data, step, index_y, n_particles, n_threads,
                             n_generators, seed);

  SEXP r_ptr = PROTECT(R_MakeExternalPtr(d, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(r_ptr, {{name}}_finalise);
  UNPROTECT(1);
  return r_ptr;
}

extern "C" SEXP {{name}}_run(SEXP ptr, SEXP r_step_end) {
  const size_t step_end = dust::util::as_size(r_step_end, "step_end");

  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);
  obj->run(step_end);

  const size_t n_state = obj->n_state();
  const size_t n_particles = obj->n_particles();

  std::vector<double> dat(n_state * n_particles);
  obj->state(dat);

  SEXP ret = PROTECT(allocMatrix(REALSXP, n_state, n_particles));
  memcpy(REAL(ret), dat.data(), dat.size() * sizeof(double));
  UNPROTECT(1);

  return ret;
}

extern "C" SEXP {{name}}_reset(SEXP ptr, SEXP data, SEXP r_step) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);
  size_t step = dust::util::as_size(r_step, "step");
  obj->reset(data, step);
  return R_NilValue;
}


extern "C" SEXP {{name}}_state(SEXP ptr) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);

  const size_t n_state = obj->n_state_full();
  const size_t n_particles = obj->n_particles();

  std::vector<double> dat(n_state * n_particles);
  obj->state_full(dat);

  SEXP ret = PROTECT(allocMatrix(REALSXP, n_state, n_particles));
  memcpy(REAL(ret), dat.data(), dat.size() * sizeof(double));
  UNPROTECT(1);

  return ret;
}


extern "C" SEXP {{name}}_step(SEXP ptr) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);
  return ScalarInteger(obj->step());
}


extern "C" SEXP {{name}}_reorder(SEXP ptr, SEXP r_index) {
  dust::Dust<{{type}}> *obj =
    dust::util::read_r_pointer<dust::Dust<{{type}}>>(ptr, true);
  size_t n = obj->n_particles();
  if (length(r_index) != n) {
    Rf_error("Expected a vector of length %d for 'index'", n);
  }
  int * r_index_data = INTEGER(r_index);

  // There are two ways of doing the conversion between base-1 and
  // base-0; we can do the offset here (as we do) or we could do it in
  // the Dust::reorder method.
  //
  // I've opted to do the conversion here (and from int to size_t too)
  // as that keeps the C++ interface tidy, which has made it easier to
  // reason about generally.  All the R bits work in native R types
  // and base-1 and all the C++ bits work in C++ types and base-0.
  bool ok = true;
  {
    std::vector<size_t> index;
    index.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      int x = r_index_data[i];
      if (x < 1 || x > n) {
        ok = false;
        break;
      }
      index.push_back(x - 1);
    }

    if (ok) {
      obj->reorder(index);
    }
  }

  if (!ok) {
    // The std::vector will have been destructed by this point due to
    // the scope above so it is safe to throw again.  The other way of
    // doing this would be to go through the array twice; once to
    // validate and the other to create the C++ index.
    Rf_error("All elements of 'index' must lie in [1, %d]", n);
  }

  return R_NilValue;
}
