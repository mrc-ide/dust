#include <R.h>
#include <Rinternals.h>

typedef void* model_create(SEXP user);
typedef void model_update(void *data, size_t step, const double *y,
                          double *ynext);
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
  size_t n_y;
  size_t n_index_y;
  particle * particles;
} dust;

void particle_init(model_create* f_create, model_update * f_update,
                   model_free *f_free,
                   particle* obj, size_t n_y, double *y, SEXP user,
                   size_t n_index_y, size_t *index_y);
particle* particle_alloc(model_create* f_create, model_update * f_update,
                         model_free *f_free,
                         size_t n_y, double *y, SEXP user,
                         size_t n_index_y, size_t *index_y);
void particle_free(particle* obj);
void particle_run(particle *obj, size_t step_end);
void particle_copy_state(particle *obj, double *dest);

dust* dust_alloc(model_create* f_create, model_update * f_update,
                 model_free *f_free,
                 size_t n_particles, size_t n_y, double *y, SEXP user,
                 size_t n_index_y, size_t *index_y);
void dust_free(dust* obj);
void dust_run(dust *obj, size_t step_end);
void dust_copy_state(dust *obj, double *ret);
