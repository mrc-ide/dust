#include <R.h>
#include <Rinternals.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef void* RNG;
typedef void* model_create(SEXP user);
typedef void model_update(void *data, size_t step, const double *y,
                          RNG *rng, size_t thread_idx, double *ynext);
typedef void model_free(void *data);

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
  size_t model_rng_calls;
  size_t n_y;
  size_t n_index_y;
  particle * particles;
  RNG* rng;
} dust;

// Prototypes for C++ calls
RNG* C_RNG_alloc(const size_t n_threads);
void C_RNG_free(RNG* obj) ;
void C_jump(RNG*, size_t, size_t);
int C_rbinom(RNG*, size_t, double, int);
int C_rbinom_tf(RNG*, size_t, double, int);
double C_rnorm(RNG*, size_t, double, double);

void particle_init(model_create* f_create, model_update * f_update,
                   model_free *f_free,
                   particle* obj, size_t n_y, double *y, SEXP user,
                   size_t n_index_y, size_t *index_y);
particle* particle_alloc(model_create* f_create, model_update * f_update,
                         model_free *f_free,
                         size_t n_y, double *y, SEXP user,
                         size_t n_index_y, size_t *index_y);
void particle_free(particle* obj);
void particle_run(particle *obj, size_t step_end, RNG *rng, size_t thread_idx);
void particle_copy_state(particle *obj, double *dest);

dust* dust_alloc(model_create* f_create, model_update * f_update,
                 model_free *f_free,
                 size_t n_particles, size_t n_threads, size_t model_rng_calls, 
                 size_t n_y, double *y, SEXP user,
                 size_t n_index_y, size_t *index_y);
void dust_free(dust* obj);
void dust_run(dust *obj, size_t step_end);
void dust_copy_state(dust *obj, double *ret);
