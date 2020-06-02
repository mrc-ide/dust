context("particle")

test_that("simulation with one particle", {
  ## TODO: We might write a reference that does not use odin here;
  ## this is not complicated stuff.
  gen <- odin::odin({
    ## nolint start
    update(S) <- S - n_SI
    update(I) <- I + n_SI - n_IR
    update(R) <- R + n_IR

    initial(S) <- S0
    initial(I) <- I0
    initial(R) <- 0

    p_SI <- 1 - exp(-beta * I / N)
    p_IR <- 1 - exp(-gamma)

    n_SI <- rbinom(S, p_SI * dt)
    n_IR <- rbinom(I, p_IR * dt)

    steps_per_day <- user(4)
    S0 <- user(1000)
    I0 <- user(10)
    beta <- user(0.2)
    gamma <- user(0.1)

    dt <- 1 / steps_per_day
    N <- S + I + R
    ## nolint end
  }, verbose = FALSE)

  mod <- gen()
  y <- mod$initial(0)
  steps <- seq(0, 100, by = 4)

  set.seed(1)
  cmp <- mod$run(steps)

  user <- NULL
  index_y <- 2L
  p <- particle_alloc(example_sir(), y, user, index_y)
  expect_is(p, "externalptr")

  set.seed(1)

  res <- matrix(NA_real_, 26, ncol(cmp))
  s <- particle_state(p)
  res[1, 1] <- 0
  res[1, 2:4] <- s

  for (i in seq_along(steps)[-1]) {
    to <- (i - 1) * 4
    x1 <- particle_run(p, to)
    x2 <- particle_state(p)
    res[i, 1] <- particle_step(p)
    res[i, 2:4] <- x2
  }

  ## Step:
  expect_equal(res[, 1], cmp[, 1])

  ## Variables:
  expect_equal(res[, 2:4], unname(cmp[, 2:4]))

  ## Force the finaliser:
  rm(p)
  gc()
})


test_that("pointer validation", {
  y <- c(1000, 10, 0)
  user <- NULL
  index_y <- 2L
  p <- particle_alloc(example_sir(), y, user, index_y)
  expect_error(particle_step(NULL), "Expected an external pointer")
  expect_error(
    particle_step(unserialize(serialize(p, NULL))),
    "Pointer has been invalidated (perhaps serialised?)",
    fixed = TRUE)
  model <- example_sir()
  model$create <- unserialize(serialize(model$create, NULL))
  expect_error(
    particle_alloc(model, y, user, index_y),
    "Function pointer has been invalidated (perhaps serialised?)",
    fixed = TRUE)
})
