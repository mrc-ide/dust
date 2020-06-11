context("example")

test_that("create walk, stepping for one step", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  obj <- res$new(1, 0, 10, 1)
  y <- obj$run(1)
  cmp <- .Call(Ctest_rng, 10L, 1L)
  expect_identical(drop(y), cmp)
})


test_that("walk agrees with random number stream", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  obj <- res$new(1, 0, 10, 1)
  y <- obj$run(5)

  cmp <- .Call(Ctest_rng, 50L, 1L)
  expect_equal(drop(y), colSums(matrix(cmp, 5, 10)))
  expect_identical(obj$state(), y)
})


test_that("Reset particles and resume continues with rng", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  sd1 <- 2
  sd2 <- 4

  obj <- res$new(sd1, 0, 10, 1)
  y1 <- obj$run(5)
  expect_equal(obj$step(), 5)
  obj$reset(sd2, 0)
  expect_equal(obj$step(), 0)
  y2 <- obj$run(5)
  expect_equal(obj$step(), 5)

  ## Then draw the random numbers:
  cmp <- .Call(Ctest_rng, 100L, 1L)
  m1 <- matrix(cmp[1:50], 5, 10)
  m2 <- matrix(cmp[51:100], 5, 10)

  expect_equal(drop(y1), colSums(m1) * sd1)
  expect_equal(drop(y2), colSums(m2) * sd2)
})
