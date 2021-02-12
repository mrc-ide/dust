example_filter <- function() {
  model <- dust_example("sir")
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


example_volatility <- function(pars = NULL) {
  pars <- pars %||% list(alpha = 0.91, sigma = 1, gamma = 1, tau = 1)

  kalman_filter <- function(pars, data) {
    alpha <- pars$alpha
    sigma <- pars$sigma
    gamma <- pars$gamma
    tau <- pars$tau
    y <- data$observed

    mu <- 0
    s <- 1
    log_likelihood <- 0

    for (t in seq_along(y)) {
      mu <- alpha * mu
      s <- alpha^2 * s + sigma^2
      m <- gamma * mu

      S <- gamma^2 * s + tau^2 # nolint
      K <- gamma * s / S # nolint

      mu <- mu + K * (y[t] - m)
      s <- s - gamma * K * s

      log_likelihood <- log_likelihood + dnorm(y[t], m, sqrt(S), log = TRUE)
    }

    log_likelihood
  }

  set.seed(1) # random for init and obs
  mod <- volatility$new(list(alpha = 0.91, sigma = 1), 0, 1L, seed = 1L)
  mod$set_state(matrix(rnorm(1L, 0, 1L), 1))
  steps <- seq(0, 100, by = 1)
  res <- dust::dust_iterate(mod, steps)
  observed <- res[1, 1, -1] + rnorm(length(steps) - 1, 0, 1)
  data <- data.frame(step = steps[-1], observed = observed)

  compare <- function(state, observed, pars) {
    dnorm(observed$observed, pars$gamma * drop(state), pars$tau, log = TRUE)
  }

  list(pars = pars, data = data, compare = compare,
       kalman_filter = kalman_filter)
}
