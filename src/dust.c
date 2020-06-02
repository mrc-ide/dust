#include "dust.h"

rng_array* rng_init(size_t n_generators,
                    const gsl_rng_type* gen_type,
                    unsigned long seed) {
  rng_array * rng = (rng_array*) calloc(1, sizeof(rng_array));
  rng->seed = seed;
  rng->gen_type = gen_type;
  rng->n_generators = n_generators;

  rng->generators = (gsl_rng**) calloc(n_generators, sizeof(gsl_rng*));
  if (rng->generators) {
    gsl_set_error_handler_off();
    for (size_t gen_idx = 0; gen_idx < n_generators; gen_idx++) {
      *(rng->generators + gen_idx) = gsl_rng_alloc(gen_type);
      if (*(rng->generators + gen_idx) == NULL) {
        Rf_error("Could not allocate memory for RNGs");
      }
      gsl_rng_set(*(rng->generators + gen_idx), rng->seed + gen_idx);
    }
  } else {
    Rf_error("Could not allocate memory for RNGs");
  }
  return(rng);
}

void rng_free(rng_array* rngs) {
  for (size_t gen_idx = 0; gen_idx < rngs->n_generators; gen_idx++) {
    gsl_rng_free(*(rngs->generators + gen_idx));
  }
  free(rngs->generators);
  free(rngs);
}

void particle_init(model_create* f_create, model_update * f_update,
                   model_free *f_free,
                   particle* obj, size_t n_y, double *y, SEXP user,
                   size_t n_index_y, size_t *index_y) {
  // TODO: this is problematic because it's outside of the smart
  // pointer control
  obj->data = (void*) (*f_create)(user);
  obj->step = 0;
  obj->n_y = n_y;
  obj->y = (double*) Calloc(n_y, double);
  obj->y_swap = (double*) Calloc(n_y, double);
  obj->update = f_update;
  obj->free = f_free;

  obj->n_index_y = n_index_y;
  obj->index_y = (size_t*) Calloc(n_index_y, size_t);
  for (size_t i = 0; i < n_index_y; ++i) {
    obj->index_y[i] = index_y[i];
  }

  memcpy(obj->y, y, n_y * sizeof(double));
}


particle* particle_alloc(model_create* f_create, model_update * f_update,
                         model_free *f_free,
                         size_t n_y, double *y, SEXP user,
                         size_t n_index_y, size_t *index_y) {
  particle *obj = (particle*) Calloc(1, particle);
  particle_init(f_create, f_update, f_free, obj, n_y, y, user,
                n_index_y, index_y);
  return obj;
}


void particle_free(particle* obj) {
  if (obj) {
    obj->free(obj->data);
    Free(obj->y);
    Free(obj->y_swap);
    Free(obj);
  }
}

void particle_run(particle *obj, size_t step_end, gsl_rng *rng) {
  while (obj->step < step_end) {
    obj->update(obj->data, obj->step, obj->y, rng, obj->y_swap);
    obj->step++;
    double *y_tmp = obj->y;
    obj->y = obj->y_swap;
    obj->y_swap = y_tmp;
  }
}


void particle_copy_state(particle *obj, double *dest) {
  for (size_t i = 0; i < obj->n_index_y; ++i) {
    dest[i] = obj->y[obj->index_y[i]];
  }
}

dust* dust_alloc(model_create* f_create, model_update * f_update,
                 model_free *f_free,
                 size_t n_particles, size_t n_threads, size_t n_y, double *y, SEXP user,
                 size_t n_index_y, size_t *index_y) {
  dust *obj = (dust*) Calloc(1, dust);
  obj->n_particles = n_particles;
  obj->n_threads = n_threads;
  obj->n_y = n_y;
  obj->n_index_y = n_index_y;
  obj->rngs = rng_init(n_threads, gsl_rng_taus2, gsl_rng_default_seed);
  obj->particles = (particle*) Calloc(n_particles, particle);
  for (size_t i = 0; i < n_particles; ++i) {
    particle_init(f_create, f_update, f_free,
                  obj->particles + i, n_y, y, user, n_index_y, index_y);
  }
  return obj;
}

void dust_free(dust* obj) {
  if (obj) {
    for (size_t i = 0; i < obj->n_particles; ++i) {
      particle *x = obj->particles + i;
      x->free(x->data);
      Free(x->y);
      Free(x->y_swap);
    }
    rng_free(obj->rngs);
    Free(obj);
  }
}

void dust_run(dust *obj, size_t step_end) {
  size_t i;
  #pragma omp parallel for private(i) schedule(static) num_threads(obj->n_threads)
  for (i = 0; i < obj->n_particles; ++i) {
    particle * x = obj->particles + i;

    size_t thread_idx = 0;
    #ifdef _OPENMP
    thread_idx = omp_get_thread_num();
    #endif
    particle_run(x, step_end, *(obj->rngs->generators + thread_idx));
  }
}

void dust_copy_state(dust *obj, double *ret) {
  size_t i;
  #pragma omp parallel for private(i) schedule(static) num_threads(obj->n_threads) ordered
  for (i = 0; i < obj->n_particles; ++i) {
    particle_copy_state(obj->particles + i, ret);
    ret += obj->n_index_y;
  }
}
