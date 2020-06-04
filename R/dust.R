dust_model <- function(create, update, free, dllname) {
  ret <- list(create = getNativeSymbolInfo(create, dllname)$address,
              update = getNativeSymbolInfo(update, dllname)$address,
              free = getNativeSymbolInfo(free, dllname)$address)
  class(ret) <- "dust_model"
  ret
}


particle_alloc <- function(model, y, user, index_y) {
  stopifnot(inherits(model, "dust_model"))
  .Call(Cparticle_alloc, model$create, model$update, model$free,
        as.numeric(y), user, as.integer(index_y))
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


dust_alloc <- function(model, n_particles, n_threads, n_rng_calls, y, user, index_y) {
  stopifnot(inherits(model, "dust_model"))
  .Call(Cdust_alloc, model$create, model$update, model$free,
        as.integer(n_particles), as.integer(n_threads), as.integer(n_rng_calls), 
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
