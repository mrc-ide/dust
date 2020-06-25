context("example")

test_that("create walk, stepping for one step", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 0, 10)
  y <- obj$run(1)
  cmp <- test_rng_norm(10, 1, 1)
  expect_identical(drop(y), cmp)
})


test_that("walk agrees with random number stream", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y <- obj$run(5)

  cmp <- test_rng_norm(50L, 1L, 1L)
  expect_equal(drop(y), colSums(matrix(cmp, 5, 10)))
  expect_identical(obj$state(), y)
})


test_that("Reset particles and resume continues with rng", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  sd1 <- 2
  sd2 <- 4

  obj <- res$new(list(sd = sd1), 0, 10)
  expect_equal(obj$info(), sd1)
  y1 <- obj$run(5)
  expect_equal(obj$step(), 5)
  obj$reset(list(sd = sd2), 0)
  expect_equal(obj$info(), sd2)
  expect_equal(obj$step(), 0)
  y2 <- obj$run(5)
  expect_equal(obj$step(), 5)

  ## Then draw the random numbers:
  cmp <- test_rng_norm(100L, 1L, 1L)
  m1 <- matrix(cmp[1:50], 5, 10)
  m2 <- matrix(cmp[51:100], 5, 10)

  expect_equal(drop(y1), colSums(m1) * sd1)
  expect_equal(drop(y2), colSums(m2) * sd2)
})


test_that("Basic sir model", {
  res <- compile_and_load(dust_file("examples/sir.cpp"), "sir", "mysir",
                          quiet = TRUE)

  obj <- res$new(list(), 0, 100)
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

  expect_equal(obj$info(), c("S", "I", "R"))
})


test_that("reorder", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)

  obj <- res$new(list(sd = 1), 0, 10)
  y1 <- obj$run(5)

  ## Simplest permutation:
  index <- rev(seq_along(y1))

  obj$reorder(index)
  y2 <- obj$state()
  expect_equal(drop(y2), rev(y1))

  y3 <- obj$run(10)

  cmp <- test_rng_norm(100L, 1L, 1L)
  m1 <- matrix(cmp[1:50], 5, 10)
  m2 <- m1[, index]
  m3 <- matrix(cmp[51:100], 5, 10)

  expect_equal(drop(y1), colSums(m1))
  expect_equal(drop(y2), colSums(m2))
  expect_equal(drop(y3), colSums(rbind(m2, m3)))
})


test_that("reorder and duplicate", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)

  obj <- res$new(list(sd = 1), 0, 10)
  y1 <- obj$run(5)

  index <- c(1L, 4L, 9L, 7L, 7L, 2L, 5L, 9L, 9L, 5L)

  obj$reorder(index)
  y2 <- obj$state()
  expect_equal(drop(y2), y1[index])

  y3 <- obj$run(10)

  cmp <- test_rng_norm(100L, 1L, 1L)
  m1 <- matrix(cmp[1:50], 5, 10)
  m2 <- m1[, index]
  m3 <- matrix(cmp[51:100], 5, 10)

  expect_equal(drop(y1), colSums(m1))
  expect_equal(drop(y2), colSums(m2))
  expect_equal(drop(y3), colSums(rbind(m2, m3)))
})


test_that("validate reorder vector is correct length", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 0, 10)
  expect_error(obj$reorder(integer(0)),
               "Expected a vector of length 10 for 'index'")
  expect_error(obj$reorder(integer(100)),
               "Expected a vector of length 10 for 'index'")
})


test_that("validate reorder vector is in correct range", {
  res <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                          quiet = TRUE)
  obj <- res$new(list(sd = 1), 0, 10)
  index <- seq_len(10)
  msg <- "All elements of 'index' must lie in [1, 10]"

  index[5] <- 0L
  expect_error(obj$reorder(index), msg, fixed = TRUE)

  index[5] <- 11
  expect_error(obj$reorder(index), msg, fixed = TRUE)

  index[5] <- -1L
  expect_error(obj$reorder(index), msg, fixed = TRUE)
})


test_that("run in float mode", {
  res_d <- compile_and_load(dust_file("examples/walk.cpp"), "walk", "mywalk",
                            quiet = TRUE)

  path <- tempfile(fileext = ".cpp")
  code <- readLines(dust_file("examples/walk.cpp"))
  pat <- "typedef double real_t"
  stopifnot(sum(grepl(pat, code)) == 1)
  writeLines(sub(pat, "typedef float real_t", code), path)
  res_f <- compile_and_load(path, "walk", "mywalkf", quiet = TRUE)

  n <- 1000
  obj_d <- res_d$new(list(sd = 10), 0, n)
  obj_f <- res_f$new(list(sd = 10), 0, n)

  y_d <- obj_d$run(10)
  y_f <- obj_f$run(10)

  expect_equal(y_d, y_f, tolerance = 1e-5)
  expect_false(identical(y_d, y_f))
})
