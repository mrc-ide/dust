context("example")

test_that("create walk, stepping for one step", {
  p <- .Call(Ctest_walk_alloc, 1, 10L, 1L)
  expect_is(p, "externalptr")

  res <- .Call(Ctest_walk_run, p, 1L)
  cmp <- .Call(Ctest_rng, 10L, 1L)
  expect_identical(drop(res), cmp)
})


test_that("walk agrees with random number stream", {
  p <- .Call(Ctest_walk_alloc, 1, 10L, 1L)
  expect_is(p, "externalptr")

  res <- .Call(Ctest_walk_run, p, 5L)
  cmp <- .Call(Ctest_rng, 50L, 1L)
  expect_equal(drop(res), colSums(matrix(cmp, 5, 10)))
})


test_that("Create object from external example file", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  obj <- res$new(1, 10, 1)
  y <- obj$run(5)

  cmp <- .Call(Ctest_rng, 50L, 1L)
  expect_equal(drop(y), colSums(matrix(cmp, 5, 10)))
})


test_that("Reset particles and resume continues with rng", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  sd1 <- 2
  sd2 <- 4

  obj <- res$new(sd1, 10, 1)
  y1 <- obj$run(5)
  obj$reset(sd2)
  y2 <- obj$run(5)

  ## Then draw the random numbers:
  cmp <- .Call(Ctest_rng, 100L, 1L)
  m1 <- matrix(cmp[1:50], 5, 10)
  m2 <- matrix(cmp[51:100], 5, 10)

  expect_equal(drop(y1), colSums(m1) * sd1)
  expect_equal(drop(y2), colSums(m2) * sd2)
})
