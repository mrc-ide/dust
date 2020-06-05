context("dust")

test_that("dust of size 1 gives same answer as single particle", {
  y <- c(1000, 10, 0)
  user <- NULL
  index_y <- 2L
  pp <- particle_alloc(example_sir(), y, user, index_y)
  by <- 5
  steps <- seq(0, 100, by = by)

  set.seed(1)
  seed <- 1
  cmp <- vector("list", length(steps))
  for (i in seq_along(steps)) {
    if (i == 1) {
      value <- NULL
    } else {
      value <- matrix(particle_run(pp, (i - 1) * by, seed))
    }
    cmp[[i]] <- list(step = particle_step(pp),
                     state = matrix(particle_state(pp)),
                     value = value)
  }

  set.seed(1)
  n_particles <- 1
  n_threads <- 1
  seed <- 1 
  ps <- dust_alloc(example_sir(), n_particles, n_threads, seed, y, user, index_y)
  expect_is(ps, "externalptr")

  res <- vector("list", length(steps))
  for (i in seq_along(steps)) {
    if (i == 1) {
      value <- NULL
    } else {
      value <- matrix(dust_run(ps, (i - 1) * by))
    }
    res[[i]] <- list(step = dust_step(ps),
                     state = matrix(dust_state(ps)),
                     value = value)
  }

  expect_equal(res, cmp)

  ## Force the finaliser:
  rm(ps)
  gc()
})

test_that("dust works with multiple threads", {
  y <- c(1000, 10, 0)
  user <- NULL
  index_y <- 2L
  by <- 5
  steps <- seq(0, 100, by = by)
  n_particles <- 2
  n_threads <- 2
  seed <- 1 
  
  ps <- dust_alloc(example_sir(), n_particles, n_threads, seed, y, user, index_y)
  expect_is(ps, "externalptr")
  
  res <- vector("list", length(steps))
  for (i in seq_along(steps)) {
    if (i == 1) {
      value <- NULL
    } else {
      value <- matrix(dust_run(ps, (i - 1) * by))
    }
    res[[i]] <- list(step = dust_step(ps),
                     state = matrix(dust_state(ps)),
                     value = value)
  }
  
  ## Force the finaliser:
  rm(ps)
  gc()
})

test_that("random walk", {
  y <- 0
  user <- c(0, 1)
  index_y <- 1L
  n_particles <- 1024
  n_threads <- 2
  n_rng <- 2
  pp <- dust_alloc(example_walk(), n_particles, n_threads, n_rng,
                   y, user, index_y)

  res <- helper_run_dust(50, 1, pp)
})
