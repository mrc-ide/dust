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
