particle_alloc <- function(y, user, index_y) {
  .Call(Cparticle_alloc, as.numeric(y), user, as.integer(index_y))
}


particle_run <- function(ptr, steps_end) {
  .Call(Cparticle_run, ptr, as.integer(steps_end))
}


particle_state <- function(ptr) {
  .Call(Cparticle_state, ptr)
}


particle_step <- function(ptr) {
  .Call(Cparticle_step, ptr)
}


dust_alloc <- function(n_particles, y, user, index_y) {
  .Call(Cdust_alloc, as.integer(n_particles),
        as.numeric(y), user, as.integer(index_y))
}


dust_run <- function(ptr, steps_end) {
  .Call(Cdust_run, ptr, as.integer(steps_end))
}


dust_state <- function(ptr) {
  .Call(Cdust_state, ptr)
}


dust_step <- function(ptr) {
  .Call(Cdust_step, ptr)
}
