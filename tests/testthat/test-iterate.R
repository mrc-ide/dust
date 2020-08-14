context("simulate")

test_that("simulate simple walk", {
  res <- dust_example("walk")
  ns <- 100
  np <- 10

  obj <- res$new(list(sd = 1), 0, np, seed = 1L)
  m <- dust_iterate(obj, 0:ns)
  expect_equal(dim(m), c(1, np, ns + 1))

  ## Compare against the direct version:
  rng <- dust_rng$new(1, np)
  rand <- matrix(rng$rnorm(np * ns, 0, 1), np, ns)
  expect_equal(t(rbind(0, apply(rand, 1, cumsum))),
               drop(m))
})


test_that("filter output", {
  res <- dust_example("sir")

  m1 <- dust_iterate(res$new(list(), 0, 100, seed = 1L), 0:100)
  m2 <- dust_iterate(res$new(list(), 0, 100, seed = 1L), 0:100, 2L)
  m3 <- dust_iterate(res$new(list(), 0, 100, seed = 1L), 0:100, c(1L, 3L))

  expect_equal(dim(m1), c(4, 100, 101))
  expect_equal(dim(m2), c(1, 100, 101))
  expect_equal(dim(m3), c(2, 100, 101))
  expect_identical(m2, m1[2L, , , drop = FALSE])
  expect_identical(m3, m1[c(1L, 3L), , , drop = FALSE])
})


test_that("validate start time", {
  res <- dust_example("walk")

  obj <- res$new(list(sd = 1), 4, 10, seed = 1L)
  expect_error(dust_iterate(obj, 0:100),
               "Expected first 'steps' element to be 4")
  expect_error(dust_iterate(obj, 10:100),
               "Expected first 'steps' element to be 4")
})
