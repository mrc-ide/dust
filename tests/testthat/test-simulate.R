context("simulate")


test_that("simulate trajectories with multiple starting points/parameters", {
  res <- dust_example("walk")
  ns <- 7
  np <- 13

  sd <- runif(np)
  data <- lapply(sd, function(p) list(sd = p))
  y0 <- matrix(rnorm(np), 1)

  steps <- seq(0, to = ns, by = 1L)

  mod <- res$new(list(sd = 1), 0, np, seed = 1L)

  ans <- dust_simulate(res, steps, data, y0, 1L, 1L, 1L)
  expect_equal(dim(ans), c(1, np, ns + 1L))
  expect_equal(ans[1, , 1], drop(y0))

  expect_identical(dust_simulate(mod, steps, data, y0, 1L, 1L, 1L), ans)

  cmp <- dust_iterate(mod, steps)
  expect_equal(ans[, , 2], cmp[, , 2] * sd + drop(y0))
  expect_equal(ans, cmp * sd + drop(y0))
})


test_that("simulate multi-state model", {
  res <- dust_example("sir")

  np <- 13

  data <- replicate(np, list(beta = runif(1, 0.15, 0.25),
                             alpha = runif(1, 0.05, 0.15)), simplify = FALSE)
  y0 <- matrix(c(1000, 10, 0, 0), 4, np)
  steps <- seq(0, 200, by = 20)

  ans <- dust_simulate(res, steps, data, y0, seed = 1L)

  expect_equal(dim(ans), c(4, np, length(steps)))
  ## Basic checks on the model:
  expect_true(all(diff(t(ans[1, , ])) <= 0))
  expect_true(all(diff(t(ans[3, , ])) >= 0))
  expect_true(all(apply(ans[1:3, , ], 2:3, sum) == 1010))
  expect_true(all(ans[4, , ] == 1000 - ans[1, , ]))

  ## And we can filter
  expect_equal(
    dust_simulate(res, steps, data, y0, index = 1L, seed = 1L),
    ans[1, , , drop = FALSE])
  expect_equal(
    dust_simulate(res, steps, data, y0, index = c(1L, 3L), seed = 1L),
    ans[c(1, 3), , , drop = FALSE])
})


test_that("simulate requires a compatible object", {
  expect_error(
    dust_simulate(NULL, 0:10, list(list()), matrix(1, 1)),
    "Expected a dust object or generator for 'model'")
})


test_that("simulate requires a matrix for initial state", {
  res <- dust_example("sir")
  expect_error(
    dust_simulate(res, 0:10, list(list()), 1),
    "Expected 'state' to be a matrix")
})


test_that("simulate requires that data and state are compatible", {
  res <- dust_example("sir")
  y0 <- matrix(1, 1, 5)
  data <- rep(list(list(sd = 1)), 4)

  expect_error(
    dust_simulate(res, 0:10, data, y0),
    "Expected 'state' to be a matrix with 4 columns")
})


test_that("simulate requires that particles have the same size", {
  res <- dust_example("variable")
  data <- list(list(len = 10), list(len = 9))
  y0 <- matrix(1, 10, 2)
  expect_error(
    dust_simulate(res, 0:10, data, y0),
    paste("Particles have different state sizes:",
          "particle 2 had length 9 but expected 10"))

  i <- rep(1:2, each = 4)
  expect_error(
    dust_simulate(res, 0:10, data[i], y0[, i, drop = FALSE]),
    paste("Particles have different state sizes:",
          "particle 5 had length 9 but expected 10"))
})


test_that("data must be an unnamed list", {
  res <- dust_example("variable")
  data <- list(len = 10)
  y0 <- matrix(1, 10, 1)
  expect_error(
    dust_simulate(res, 0:10, data, y0),
    "Expected 'data' to be an unnamed list")
})


test_that("two calls with seed = NULL create different results", {
  res <- dust_example("walk")
  ns <- 7
  np <- 13

  data <- lapply(runif(np), function(p) list(sd = p))
  y0 <- matrix(rnorm(np), 1)
  steps <- seq(0, to = ns, by = 1L)
  mod <- res$new(list(sd = 1), 0, np, seed = 1L)

  ans1 <- dust_simulate(res, steps, data, y0, seed = NULL)
  ans2 <- dust_simulate(res, steps, data, y0, seed = NULL)
  expect_identical(dim(ans1), dim(ans2))
  expect_false(identical(ans1, ans2))
})


test_that("steps must not be negative", {
  res <- dust_example("sir")
  y0 <- matrix(1, 1, 5)
  data <- rep(list(list(sd = 1)), 5)
  expect_error(
    dust_simulate(res, seq(-5, 10), data, y0),
    "All elements of 'steps' must be non-negative")
})


test_that("can extract final state", {
  mod <- dust_example("variable")

  np <- 13
  ny <- 5
  ns <- 11

  steps <- 0:ns

  seed <- dust_rng$new(NULL, np)$state()

  data <- rep(list(list(len = ny)), np)
  y0 <- matrix(rnorm(ny * np), ny, np)
  res1 <- dust_simulate(mod, steps, data, y0, seed = seed, return_state = TRUE)
  expect_identical(res1[, , ns + 1], attr(res1, "state"))

  res2 <- dust_simulate(mod, steps, data, y0, seed = seed,
                        index = integer(0), return_state = TRUE)
  expect_identical(attr(res2, "state"), attr(res1, "state"))
  expect_identical(attr(res2, "rng_state"), attr(res1, "rng_state"))

  cmp <- mod$new(list(len = ny), 0, np, seed = seed)
  cmp$set_state(y0)
  cmp_state <- cmp$run(ns)
  cmp_rng_state <- cmp$rng_state()
  expect_identical(cmp_state, attr(res1, "state"))
  expect_identical(cmp_rng_state, attr(res1, "rng_state"))
})
