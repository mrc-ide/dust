example_filter <- function() {
  model <- dust_example("sir2")
  np <- 10
  end <- 150 * 4
  steps <- seq(0, end, by = 4)
  ans <- dust_iterate(model$new(list(), 0, np, seed = 1L), steps)
  dat <- data.frame(step = steps[-1], incidence = ans[5, 1, -1])
  dat_dust <- dust_data(dat)
  list(model = model, dat = dat, dat_dust = dat_dust)
}


## Functions that derive from mcstate, and which act as tests against
## fiddly C++ implementations.
scale_log_weights <- function(w) {
  max_w <- max(w)
  wi <- exp(w - max_w)
  tot <- sum(wi)
  list(average = log(tot / length(w)) + max_w, weights = wi)
}


filter_history_reorder <- function(value, order) {
  index_particle <- seq_len(ncol(value))
  n_state <- nrow(value)
  n_particles <- length(index_particle)
  n_data <- ncol(order)
  idx <- matrix(NA_integer_, n_particles, n_data)
  for (i in rev(seq_len(ncol(idx)))) {
    index_particle <- idx[, i] <- order[index_particle, i]
  }
  cidx <- cbind(seq_len(n_state),
                rep(idx, each = n_state),
                rep(seq_len(n_data), each = n_state * n_particles))
  array(value[cidx], c(n_state, n_particles, n_data))
}


## The C++ version of the above is closer to this, which might be
## useful if we decide to expose getting just one particle's history
## (or some set) out of the history.
filter_history_reorder_cpp <- function(value, order) {
  n_state <- nrow(value)
  n_particles <- ncol(value)
  n_data <- ncol(order)
  index_particle <- seq_len(ncol(value))
  ret <- array(NA_real_, c(n_state, n_particles, n_data))
  for (i in rev(seq_len(ncol(order)))) {
    index_particle <- order[index_particle, i]
    ret[, , i] <- value[, index_particle, i]
  }
  ret
}
