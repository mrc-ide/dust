context("simulate")


test_that("simulate trajectories with multiple starting points/parameters", {
  res <- dust_example("walk")
  ns <- 7
  np <- 13

  sd <- runif(np)
  pars <- lapply(sd, function(p) list(sd = p))
  y0 <- matrix(rnorm(np), 1)

  steps <- seq(0, to = ns, by = 1L)

  mod <- res$new(list(sd = 1), 0, np, seed = 1L)
  expect_warning(
    ans <- dust_simulate(res, steps, pars, y0, 1L, 1L, 1L),
    "$simulate() method directly", fixed = TRUE)

  expect_equal(dim(ans), c(1, np, ns + 1L))
  expect_equal(ans[1, , 1], drop(y0))

  expect_error(
    dust_simulate(mod, steps, pars, y0, 1L, 1L, 1L),
    "dust_simulate no longer valid for dust models")

  mod2 <- res$new(pars, 0, 1L, seed = 1L, pars_multi = TRUE)
  mod2$set_state(array(y0, c(1, 1, np)))
  ans2 <- mod2$simulate2(steps)
  expect_equal(dim(ans2), c(1, 1, np, ns + 1L))
  expect_equal(c(ans2), c(ans))

  cmp <- dust_iterate(mod, steps)
  expect_equal(ans[, , 2], cmp[, , 2] * sd + drop(y0))
  expect_equal(ans, cmp * sd + drop(y0))
})


test_that("simulate multi-state model", {
  res <- dust_example("sir")

  np <- 13

  pars <- replicate(np, list(beta = runif(1, 0.15, 0.25),
                             alpha = runif(1, 0.05, 0.15)), simplify = FALSE)
  y0 <- matrix(c(1000, 10, 0, 0, 0), 5, np)
  steps <- seq(0, 200, by = 20)

  expect_warning(
    ans <- dust_simulate(res, steps, pars, y0, seed = 1L),
    "$simulate() method directly", fixed = TRUE)

  expect_equal(dim(ans), c(5, np, length(steps)))
  ## Basic checks on the model:
  expect_true(all(diff(t(ans[1, , ])) <= 0))
  expect_true(all(diff(t(ans[3, , ])) >= 0))
  expect_true(all(apply(ans[1:3, , ], 2:3, sum) == 1010))
  expect_true(all(ans[4, , ] == 1000 - ans[1, , ]))

  ## And we can filter
  expect_equal(
    suppressWarnings(dust_simulate(res, steps, pars, y0, index = 1L,
                                   seed = 1L)),
    ans[1, , , drop = FALSE])
  expect_equal(
    suppressWarnings(dust_simulate(res, steps, pars, y0, index = c(1L, 3L),
                                   seed = 1L)),
    ans[c(1, 3), , , drop = FALSE])

  mod <- res$new(pars, 0, 1L, seed = 1L, pars_multi = TRUE)
  mod$set_state(array(y0, c(5, 1, np)))
  ans2 <- mod$simulate2(steps)
  expect_equal(dim(ans2), c(5, 1, np, length(steps)))
  expect_equal(c(ans2), c(ans))
})


test_that("simulate requires a compatible object", {
  expect_error(
    dust_simulate(NULL, 0:10, list(list()), matrix(1, 1)),
    "Expected a dust generator for 'model'")
})


test_that("simulate requires a matrix for initial state", {
  res <- dust_example("sir")
  expect_error(
    suppressWarnings(dust_simulate(res, 0:10, list(list()), 1)),
    "Expected 'state' to be a matrix")
})


test_that("simulate requires that pars and state are compatible", {
  res <- dust_example("sir")
  y0 <- matrix(1, 1, 5)
  pars <- rep(list(list(sd = 1)), 4)

  expect_error(
    suppressWarnings(dust_simulate(res, 0:10, pars, y0)),
    "Expected 'state' to be a matrix with 4 columns")
})


test_that("simulate requires that particles have the same size", {
  res <- dust_example("variable")
  pars <- list(list(len = 10), list(len = 9))
  y0 <- matrix(1, 10, 2)
  expect_error(
    suppressWarnings(dust_simulate(res, 0:10, pars, y0)),
    "expected length 10 but parameter set 2 created length 9")

  i <- rep(1:2, each = 4)
  expect_error(
    suppressWarnings(dust_simulate(res, 0:10, pars[i], y0[, i, drop = FALSE])),
    "expected length 10 but parameter set 5 created length 9")
})


test_that("pars must be an unnamed list", {
  res <- dust_example("variable")
  pars <- list(len = 10)
  y0 <- matrix(1, 10, 1)
  expect_error(
    suppressWarnings(dust_simulate(res, 0:10, pars, y0)),
    "Expected 'pars' to be an unnamed list")
})


test_that("two calls with seed = NULL create different results", {
  res <- dust_example("walk")
  ns <- 7
  np <- 13

  pars <- lapply(runif(np), function(p) list(sd = p))
  y0 <- matrix(rnorm(np), 1)
  steps <- seq(0, to = ns, by = 1L)
  mod <- res$new(list(sd = 1), 0, np, seed = 1L)
  suppressWarnings({
    ans1 <- dust_simulate(res, steps, pars, y0, seed = NULL)
    ans2 <- dust_simulate(res, steps, pars, y0, seed = NULL)
  })
  expect_identical(dim(ans1), dim(ans2))
  expect_false(identical(ans1, ans2))
})


test_that("steps must not be negative", {
  res <- dust_example("sir")
  y0 <- matrix(1, 1, 5)
  pars <- rep(list(list(sd = 1)), 5)
  expect_error(
    suppressWarnings(dust_simulate(res, seq(-5, 10), pars, y0)),
    "'step' must be non-negative")
})


test_that("can extract final state", {
  mod <- dust_example("variable")

  np <- 13
  ny <- 5
  ns <- 11

  steps <- 0:ns

  seed <- dust_rng$new(NULL, np)$state()

  pars <- rep(list(list(len = ny)), np)
  y0 <- matrix(rnorm(ny * np), ny, np)
  res1 <- suppressWarnings(
    dust_simulate(mod, steps, pars, y0, seed = seed,
                  return_state = TRUE))
  expect_identical(res1[, , ns + 1], attr(res1, "state"))

  res2 <- suppressWarnings(
    dust_simulate(mod, steps, pars, y0, seed = seed,
                  index = integer(0), return_state = TRUE))
  expect_identical(attr(res2, "state"), attr(res1, "state"))
  expect_identical(attr(res2, "rng_state"), attr(res1, "rng_state"))

  cmp <- mod$new(list(len = ny), 0, np, seed = seed)
  cmp$set_state(y0)
  cmp_state <- cmp$run(ns)
  cmp_rng_state <- cmp$rng_state()
  expect_identical(cmp_state, attr(res1, "state"))
  expect_identical(cmp_rng_state, attr(res1, "rng_state"))
})


test_that("Simulate with multiple pars", {
  res <- dust_example("sir")

  np <- 13
  pars <- list(list(beta = 0.1), list(beta = 0.2))
  steps <- seq(0, 200, by = 20)

  mod <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  seed <- mod$rng_state()
  ans <- mod$simulate2(steps)

  ## Validate against single parameter model:
  i <- seq_len(length(seed) / 2)
  cmp1 <- res$new(pars[[1]], 0, np, seed = seed[i])$simulate2(steps)
  cmp2 <- res$new(pars[[2]], 0, np, seed = seed[-i])$simulate2(steps)

  expect_equal(dim(ans), c(5, np, length(pars), length(steps)))
  expect_equal(ans[, , 1, ], cmp1)
  expect_equal(ans[, , 2, ], cmp2)

  ## Can filter
  mod2 <- res$new(pars, 0, np, seed = 1L, pars_multi = TRUE)
  mod2$set_index(c(x = 4L))
  ans2 <- mod2$simulate2(steps)
  expect_equal(unname(ans2), ans[4, , , , drop = FALSE])
  expect_equal(dimnames(ans2), list("x", NULL, NULL, NULL))
})
