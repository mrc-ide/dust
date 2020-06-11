context("example")

test_that("create walk, stepping for one step", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  obj <- res$new(1, 0, 10)
  y <- obj$run(1)
  cmp <- .Call(Ctest_rng, 10L, 1L, 1L)
  expect_identical(drop(y), cmp)
})


test_that("walk agrees with random number stream", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  obj <- res$new(1, 0, 10, seed = 1L)
  y <- obj$run(5)

  cmp <- .Call(Ctest_rng, 50L, 1L, 1L)
  expect_equal(drop(y), colSums(matrix(cmp, 5, 10)))
  expect_identical(obj$state(), y)
})


test_that("Reset particles and resume continues with rng", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "my_walk")
  sd1 <- 2
  sd2 <- 4

  obj <- res$new(sd1, 0, 10)
  y1 <- obj$run(5)
  expect_equal(obj$step(), 5)
  obj$reset(sd2, 0)
  expect_equal(obj$step(), 0)
  y2 <- obj$run(5)
  expect_equal(obj$step(), 5)

  ## Then draw the random numbers:
  cmp <- .Call(Ctest_rng, 100L, 1L, 1L)
  m1 <- matrix(cmp[1:50], 5, 10)
  m2 <- matrix(cmp[51:100], 5, 10)

  expect_equal(drop(y1), colSums(m1) * sd1)
  expect_equal(drop(y2), colSums(m2) * sd2)
})


test_that("Basic sir model", {
  res <- compile_and_load(dust_file("examples/sir.cpp"), "sir", "my_sir")

  obj <- res$new(NULL, 0, 100)
  ans <- vector("list", 150)
  for (i in seq_along(ans)) {
    value <- obj$run(i * 4)
    ans[[i]] <- list(value = value, state = obj$state(), step = obj$step())
  }

  step <- vapply(ans, function(x) x$step, numeric(1))
  state_s <- t(vapply(ans, function(x) x$state[1, ], numeric(100)))
  state_i <- t(vapply(ans, function(x) x$state[2, ], numeric(100)))
  state_r <- t(vapply(ans, function(x) x$state[3, ], numeric(100)))
  value <- t(vapply(ans, function(x) drop(x$value), numeric(100)))

  n <- nrow(state_s)
  expect_true(all(state_s[-n, ] - state_s[-1, ] >= 0))
  expect_true(all(state_r[-n, ] - state_r[-1, ] <= 0))
  expect_false(all(state_i[-n, ] - state_i[-1, ] <= 0))
  expect_false(all(state_i[-n, ] - state_i[-1, ] >= 0))
  expect_identical(value, state_s)
  expect_equal(step, seq(4, by = 4, length.out = n))
})
