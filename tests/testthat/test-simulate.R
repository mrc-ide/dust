context("simulate")


test_that("simulate trajectories with multiple starting points/parameters", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  ns <- 7
  np <- 13

  sd <- runif(np)
  data <- lapply(sd, function(p) list(sd = p))
  y0 <- matrix(rnorm(np), 1)

  steps <- seq(0, to = ns, by = 1L)

  mod <- res$new(list(sd = 1), 0, np)

  ans <- dust_simulate(res, steps, data, y0, 1L, 1L, 1L)
  expect_equal(dim(ans), c(1, np, ns + 1L))
  expect_equal(ans[1, , 1], drop(y0))

  expect_identical(dust_simulate(mod, steps, data, y0, 1L, 1L, 1L), ans)

  cmp <- dust_iterate(mod, steps)
  expect_equal(ans[, , 2], cmp[, , 2] * sd + drop(y0))
  expect_equal(ans, cmp * sd + drop(y0))
})


test_that("Simulate from current point", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  ns <- 7
  np <- 13

  mod <- res$new(list(sd = 1), 0, np)
  y0 <- mod$run(10)
  r <- mod$rng_state()

  steps <- 10:20
  ans <- dust_simulate(mod, steps)

  expect_equal(dim(ans), c(1, np, length(steps)))
  expect_equal(ans[1, , 1], drop(y0))
  expect_identical(mod$state(), y0)
  expect_identical(mod$rng_state(), r)

  cmp <- dust_iterate(res$new(list(sd = 1), 10, np), steps)
  expect_equal(ans[, , 2], cmp[, , 2] + drop(y0))
  expect_equal(ans, cmp + drop(y0))
})


test_that("Simulate from current point, using new pars", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  ns <- 7
  np <- 13

  mod <- res$new(list(sd = 1), 0, np)
  y0 <- mod$run(10)
  r <- mod$rng_state()

  sd <- runif(np)
  data <- lapply(sd, function(p) list(sd = p))

  steps <- 10:20
  ans <- dust_simulate(mod, steps, data = data)

  expect_equal(dim(ans), c(1, np, length(steps)))
  expect_equal(ans[1, , 1], drop(y0))
  expect_identical(mod$state(), y0)
  expect_identical(mod$rng_state(), r)
  expect_identical(mod$data(), list(sd = 1))

  cmp <- dust_iterate(res$new(list(sd = 1), 10, np), steps)
  expect_equal(ans[, , 2], cmp[, , 2] * sd + drop(y0))
  expect_equal(ans, cmp * sd + drop(y0))
})


test_that("Simulate from current point, using new pars and starting point", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  ns <- 7
  np1 <- 13
  np2 <- 17

  mod <- res$new(list(sd = 1), 0, np1)
  y0 <- mod$run(10)
  y1 <- matrix(runif(np2), 1)

  sd <- runif(np2)
  data <- lapply(sd, function(p) list(sd = p))

  steps <- 10:20
  ans <- dust_simulate(mod, steps, data = data, state = y1)

  expect_equal(dim(ans), c(1, np2, length(steps)))
  expect_equal(ans[1, , 1], drop(y1))

  cmp <- dust_iterate(res$new(list(sd = 1), 10, np2), steps)
  expect_equal(ans[, , 2], cmp[, , 2] * sd + drop(y1))
  expect_equal(ans, cmp * sd + drop(y1))
})
