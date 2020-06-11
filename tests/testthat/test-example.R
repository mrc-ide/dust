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
