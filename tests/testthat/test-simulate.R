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


test_that("simulate multi-state model", {
  res <- dust(dust_file("examples/sir.cpp"), quiet = FALSE)

  np <- 13

  data <- replicate(np, list(beta = runif(1, 0.15, 0.25),
                             alpha = runif(1, 0.05, 0.15)), simplify = FALSE)
  y0 <- matrix(c(1000, 10, 0), 3, np)
  steps <- seq(0, 200, by = 20)

  ans <- dust_simulate(res, steps, data, y0)

  expect_equal(dim(ans), c(3, np, length(steps)))
  ## Basic checks on the model:
  expect_true(all(diff(t(ans[1, , ])) <= 0))
  expect_true(all(diff(t(ans[3, , ])) >= 0))
  expect_true(all(apply(ans, 2:3, sum) == 1010))

  ## And we can filter
  expect_equal(
    dust_simulate(res, steps, data, y0, index = 1L),
    ans[1, , , drop = FALSE])
  expect_equal(
    dust_simulate(res, steps, data, y0, index = c(1L, 3L)),
    ans[c(1, 3), , , drop = FALSE])
})
