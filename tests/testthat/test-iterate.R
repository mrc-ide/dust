context("simulate")

test_that("simulate simple walk", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  ns <- 100
  np <- 10

  obj <- res$new(list(sd = 1), 0, np)
  m <- dust_iterate(obj, 0:ns)
  expect_equal(dim(m), c(1, np, ns + 1))

  ## Compare against the direct version:
  rng <- dust_rng$new(1, np)
  rand <- matrix(rng$rnorm(np * ns, 0, 1), np, ns)
  expect_equal(t(rbind(0, apply(rand, 1, cumsum))),
               drop(m))
})


test_that("filter output", {
  res <- compile_and_load(dust_file("examples/sir.cpp"), "sir", "mysir",
                          quiet = TRUE)

  obj <- res$new(list(), 0, 100)
  m1 <- dust_iterate(res$new(list(), 0, 100), 0:100)
  m2 <- dust_iterate(res$new(list(), 0, 100), 0:100, 2L)
  m3 <- dust_iterate(res$new(list(), 0, 100), 0:100, c(1L, 3L))

  expect_equal(dim(m1), c(3, 100, 101))
  expect_equal(dim(m2), c(1, 100, 101))
  expect_equal(dim(m3), c(2, 100, 101))
  expect_identical(m2, m1[2L, , , drop = FALSE])
  expect_identical(m3, m1[c(1L, 3L), , , drop = FALSE])
})


test_that("validate start time", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 4, 10)
  expect_error(dust_iterate(obj, 0:100),
               "Expected first 'steps' element to be 4")
  expect_error(dust_iterate(obj, 10:100),
               "Expected first 'steps' element to be 4")
})


test_that("simulate trajectories with multiple starting points/parameters", {
  res <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)
  ns <- 7
  np <- 13

  sd <- runif(np)
  data <- lapply(sd, function(p) list(sd = p))
  y0 <- matrix(rnorm(np), 1)

  steps <- seq(0, to = ns, by = 1L)

  ans <- res$parent_env$dust_walk_simulate(steps, data, y0, 1L, 1L, 1L)
  expect_equal(dim(ans), c(1, np, ns + 1L))
  expect_equal(ans[1, , 1], drop(y0))

  cmp <- dust_iterate(res$new(list(sd = 1), 0, np), steps)
  expect_equal(ans[, , 2], cmp[, , 2] * sd + drop(y0), )
  expect_equal(ans, cmp * sd + drop(y0))
})
