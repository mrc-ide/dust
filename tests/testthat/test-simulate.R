context("simulate")


test_that("simulate trajectories with multiple starting points/parameters", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
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
  res <- dust(dust_file("examples/sir.cpp"), quiet = TRUE)

  np <- 13

  data <- replicate(np, list(beta = runif(1, 0.15, 0.25),
                             alpha = runif(1, 0.05, 0.15)), simplify = FALSE)
  y0 <- matrix(c(1000, 10, 0), 3, np)
  steps <- seq(0, 200, by = 20)

  ans <- dust_simulate(res, steps, data, y0, seed = 1L)

  expect_equal(dim(ans), c(3, np, length(steps)))
  ## Basic checks on the model:
  expect_true(all(diff(t(ans[1, , ])) <= 0))
  expect_true(all(diff(t(ans[3, , ])) >= 0))
  expect_true(all(apply(ans, 2:3, sum) == 1010))

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
  res <- dust(dust_file("examples/sir.cpp"), quiet = TRUE)
  expect_error(
    dust_simulate(res, 0:10, list(list()), 1),
    "Expected 'state' to be a matrix")
})


test_that("simulate requires that data and state are compatible", {
  res <- dust(dust_file("examples/sir.cpp"), quiet = TRUE)
  y0 <- matrix(1, 1, 5)
  data <- rep(list(list(sd = 1)), 4)

  expect_error(
    dust_simulate(res, 0:10, data, y0),
    "Expected 'state' to be a matrix with 4 columns")
})


test_that("simulate requires that particles have the same size", {
  res <- dust(dust_file("examples/variable.cpp"), quiet = TRUE)
  data <- list(list(len = 10), list(len = 9))
  y0 <- matrix(1, 10, 2)
  expect_error(
    dust_simulate(res, 0:10, data, y0),
    paste("Particles have different state sizes:",
          "particle 2 had length 10 but expected 9"))

  i <- rep(1:2, each = 4)
  expect_error(
    dust_simulate(res, 0:10, data[i], y0[, i, drop = FALSE]),
    paste("Particles have different state sizes:",
          "particle 5 had length 10 but expected 9"))
})


test_that("data must be an unnamed list", {
  res <- dust(dust_file("examples/variable.cpp"), quiet = TRUE)
  data <- list(len = 10)
  y0 <- matrix(1, 10, 1)
  expect_error(
    dust_simulate(res, 0:10, data, y0),
    "Expected 'data' to be an unnamed list")
})


test_that("two calls with seed = NULL create different results", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
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
