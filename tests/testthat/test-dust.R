context("dust")

test_that("dust of size 1 gives same answer as single particle", {
  y <- c(1000, 10, 0)
  user <- NULL
  index_y <- 2L
  pp <- particle_alloc(example_sir(), y, user, index_y)
  by <- 5
  steps <- seq(0, 100, by = by)

  set.seed(1)
  cmp <- vector("list", length(steps))
  for (i in seq_along(steps)) {
    if (i == 1) {
      value <- NULL
    } else {
      value <- matrix(particle_run(pp, (i - 1) * by))
    }
    cmp[[i]] <- list(step = particle_step(pp),
                     state = matrix(particle_state(pp)),
                     value = value)
  }

  set.seed(1)
  n_particles <- 1
  n_threads <- 1
  n_rng <- 2 # Needs to be automated
  ps <- dust_alloc(example_sir(), n_particles, n_threads, n_rng, y, user, index_y)
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
  n_rng <- 2 # Needs to be automated
  
  ps <- dust_alloc(example_sir(), n_particles, n_threads, n_rng, y, user, index_y)
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
