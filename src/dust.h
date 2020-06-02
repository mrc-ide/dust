#include <R.h>
#include <Rinternals.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef void* model_create(SEXP user);
typedef void model_update(void *data, size_t step, const double *y,
                          gsl_rng *rng, double *ynext);
typedef void model_free(void *data);

typedef struct {
  gsl_rng** generators;
  unsigned long seed;
  const gsl_rng_type* gen_type;
  size_t n_generators;
} rng_array;

typedef struct {
  void * data;
  model_update * update;
  model_free * free;

  size_t step;
  size_t n_y;

  double *y;
  double *y_swap;

  size_t n_index_y;
  size_t *index_y;
} particle;

typedef struct {
  size_t n_particles;
  size_t n_threads;
  size_t n_y;
  size_t n_index_y;
  particle * particles;
  rng_array * rngs;
} dust;

rng_array* rng_init(size_t n_generators,
                    const gsl_rng_type* gen_type,
                    unsigned long seed);
void rng_free(rng_array* rngs);

void particle_init(model_create* f_create, model_update * f_update,
                   model_free *f_free,
                   particle* obj, size_t n_y, double *y, SEXP user,
                   size_t n_index_y, size_t *index_y);
particle* particle_alloc(model_create* f_create, model_update * f_update,
                         model_free *f_free,
                         size_t n_y, double *y, SEXP user,
                         size_t n_index_y, size_t *index_y);
void particle_free(particle* obj);
void particle_run(particle *obj, size_t step_end, gsl_rng *rng);
void particle_copy_state(particle *obj, double *dest);

dust* dust_alloc(model_create* f_create, model_update * f_update,
                 model_free *f_free,
                 size_t n_particles, size_t nthreads, size_t n_y, double *y, SEXP user,
                 size_t n_index_y, size_t *index_y);
void dust_free(dust* obj);
void dust_run(dust *obj, size_t step_end);
void dust_copy_state(dust *obj, double *ret);
