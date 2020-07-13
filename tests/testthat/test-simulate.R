context("simulate")

test_that("simulate simple walk", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 0, 10)
  m <- dust_simulate(obj, 0:100)
  expect_equal(dim(m), c(1, 10, 101))

  ## Compare against the direct version:
  rand <- matrix(dust_rng$new(1)$rnorm(10 * 100, 0, 1), 10, 100)
  expect_equal(t(rbind(0, apply(rand, 1, cumsum))),
               drop(m))
})


test_that("filter output", {
  res <- compile_and_load(dust_file("examples/sir.cpp"), "sir", "mysir",
                          quiet = TRUE)

  obj <- res$new(list(), 0, 100)
  m1 <- dust_simulate(res$new(list(), 0, 100), 0:100)
  m2 <- dust_simulate(res$new(list(), 0, 100), 0:100, 2L)
  m3 <- dust_simulate(res$new(list(), 0, 100), 0:100, c(1L, 3L))

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
  expect_error(dust_simulate(obj, 0:100),
               "Expected first 'steps' element to be 4")
  expect_error(dust_simulate(obj, 10:100),
               "Expected first 'steps' element to be 4")
})
