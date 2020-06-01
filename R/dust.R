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


swarm_alloc <- function(n_particles, n_y, user, index_y) {
  .Call(Cswarm_alloc, as.integer(n_particles),
        as.numeric(y), user, as.integer(index_y))
}


swarm_run <- function(ptr, steps_end) {
  .Call(Cswarm_run, ptr, as.integer(steps_end))
}


swarm_state <- function(ptr) {
  .Call(Cswarm_state, ptr)
}


swarm_step <- function(ptr) {
  .Call(Cswarm_step, ptr)
}
