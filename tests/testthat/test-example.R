context("example")

test_that("create walk, stepping for one step", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  expect_null(obj$info())

  y <- obj$run(1)
  cmp <- dust_rng$new(1, 10)$rnorm(10, 0, 1)
  expect_identical(drop(y), cmp)
})


test_that("walk agrees with random number stream", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y <- obj$run(5)

  cmp <- dust_rng$new(1, 10)$rnorm(50, 0, 1)
  expect_equal(drop(y), colSums(matrix(cmp, 5, 10, byrow = TRUE)))
  expect_identical(obj$state(), y)
})


test_that("Reset particles and resume continues with rng", {
  res <- dust_example("walk")
  sd1 <- 2
  sd2 <- 4

  obj <- res$new(list(sd = sd1), 0, 10, seed = 1L)

  y1 <- obj$run(5)
  expect_equal(obj$step(), 5)
  obj$reset(list(sd = sd2), 0)
  expect_equal(obj$step(), 0)
  y2 <- obj$run(5)
  expect_equal(obj$step(), 5)

  ## Then draw the random numbers:
  cmp <- dust_rng$new(1, 10)$rnorm(100, 0, 1)
  m1 <- matrix(cmp[1:50], 5, 10, byrow = TRUE)
  m2 <- matrix(cmp[51:100], 5, 10, byrow = TRUE)

  expect_equal(drop(y1), colSums(m1) * sd1)
  expect_equal(drop(y2), colSums(m2) * sd2)
})


test_that("Basic sir model", {
  res <- dust_example("sir")

  obj <- res$new(list(), 0, 100, seed = 1L)
  obj$set_index(1L)

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

  s <- ans[[150]]$state
  expect_equal(obj$state(), s)
  expect_equal(obj$state(1L), s[1, , drop = FALSE])
  expect_equal(obj$state(3:1), s[3:1, , drop = FALSE])
  expect_equal(obj$state(c(2L, 2L)), s[c(2, 2), , drop = FALSE])
  expect_error(obj$state(0L),
               "All elements of 'index' must lie in [1, 4]",
               fixed = TRUE)
  expect_error(obj$state(1:5),
               "All elements of 'index' must lie in [1, 4]",
               fixed = TRUE)
})


test_that("set index", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 1, seed = 1L)
  expect_equal(mod$state(), matrix(1:10))
  expect_equal(mod$run(0), matrix(1:10, nrow = 10, ncol = 1))

  mod$set_index(2:4)
  expect_equal(mod$run(0), matrix(2:4))

  y <- mod$run(1)
  expect_equal(y, mod$state(2:4))
  expect_equal(y, mod$state()[2:4, , drop = FALSE])

  mod$set_index(integer(0))
  expect_equal(mod$run(1), matrix(numeric(0), nrow = 0, ncol = 1))
})


test_that("reset clears the index", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 1, seed = 1L)
  mod$set_index(2:4)
  expect_equal(mod$run(0), matrix(2:4))
  mod$reset(list(len = 10), 0)
  expect_equal(mod$run(0), matrix(1:10, 10, 1))
})


test_that("set model state", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 1, seed = 1L)
  expect_equal(mod$state(), matrix(1:10))
  x <- runif(10)
  mod$set_state(x)
  expect_equal(mod$state(), matrix(x))
  expect_error(
    mod$set_state(1),
    "Expected a vector with 10 elements for 'state'")
  expect_equal(mod$state(), matrix(x))
})


test_that("set model state into multiple particles", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 20, seed = 1L)
  expect_equal(mod$state(), matrix(1:10, 10, 20))
  x <- runif(10)
  mod$set_state(x)
  expect_equal(mod$state(), matrix(x, 10, 20))
  expect_error(
    mod$set_state(1),
    "Expected a vector with 10 elements for 'state'")
  expect_equal(mod$state(), matrix(x, 10, 20))
})


test_that("set model state with a matrix", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(runif(20), 10, 2)
  mod$set_state(m)

  expect_equal(mod$state(), m)
  expect_error(
    mod$set_state(m[, c(1, 1, 2, 2)]),
    "Expected a matrix with 2 columns for 'state'")
  expect_error(
    mod$set_state(m[1:5, ]),
    "Expected a matrix with 10 rows for 'state'")
  expect_equal(mod$state(), m)
})


test_that("reorder", {
  res <- dust_example("walk")

  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y1 <- obj$run(5)

  ## Simplest permutation:
  index <- rev(seq_along(y1))

  obj$reorder(index)
  y2 <- obj$state()
  expect_equal(drop(y2), rev(y1))

  y3 <- obj$run(10)

  cmp <- dust_rng$new(1, 10)$rnorm(100, 0, 1)
  m1 <- matrix(cmp[1:50], 5, 10, byrow = TRUE)
  m2 <- m1[, index]
  m3 <- matrix(cmp[51:100], 5, 10, byrow = TRUE)

  expect_equal(drop(y1), colSums(m1))
  expect_equal(drop(y2), colSums(m2))
  expect_equal(drop(y3), colSums(rbind(m2, m3)))
})


test_that("reorder and duplicate", {
  res <- dust_example("walk")

  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  y1 <- obj$run(5)

  index <- c(1L, 4L, 9L, 7L, 7L, 2L, 5L, 9L, 9L, 5L)

  obj$reorder(index)
  y2 <- obj$state()
  expect_equal(drop(y2), y1[index])

  y3 <- obj$run(10)

  cmp <- dust_rng$new(1, 10)$rnorm(100, 0, 1)
  m1 <- matrix(cmp[1:50], 5, 10, byrow = TRUE)
  m2 <- m1[, index]
  m3 <- matrix(cmp[51:100], 5, 10, byrow = TRUE)

  expect_equal(drop(y1), colSums(m1))
  expect_equal(drop(y2), colSums(m2))
  expect_equal(drop(y3), colSums(rbind(m2, m3)))
})


test_that("validate reorder vector is correct length", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
  expect_error(obj$reorder(1L),
               "Expected a vector of length 10 for 'index'")
  expect_error(obj$reorder(rep(1L, 100)),
               "Expected a vector of length 10 for 'index'")
})


test_that("validate reorder vector is in correct range", {
  res <- dust_example("walk")
  obj <- res$new(list(sd = 1), 0, 10, seed = 1L)
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
  res_d <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)

  path <- tempfile(fileext = ".cpp")
  code <- readLines(dust_file("examples/walk.cpp"))
  pat <- "typedef double real_t"
  stopifnot(sum(grepl(pat, code)) == 1)
  writeLines(sub(pat, "typedef float real_t", code), path)
  res_f <- dust(path, quiet = TRUE)

  n <- 1000
  obj_d <- res_d$new(list(sd = 10), 0, n, seed = 1L)
  obj_f <- res_f$new(list(sd = 10), 0, n, seed = 1L)

  y_d <- obj_d$run(10)
  y_f <- obj_f$run(10)

  expect_equal(y_d, y_f, tolerance = 1e-5)
  expect_false(identical(y_d, y_f))
})


test_that("reset changes info", {
  res <- dust_example("sir")
  obj <- res$new(list(), 0, 100, seed = 1L)
  expect_equal(obj$info(),
               list(vars = c("S", "I", "R", "inc"),
                    pars = list(beta = 0.2, gamma = 0.1)))
  obj$reset(list(beta = 0.1), 0)
  expect_equal(obj$info(),
               list(vars = c("S", "I", "R", "inc"),
                    pars = list(beta = 0.1, gamma = 0.1)))
})


test_that("Basic threading test", {
  res <- dust(dust_file("examples/parallel.cpp"), quiet = TRUE)

  obj <- res$new(list(sd = 1), 0, 10, n_threads = 2L, seed = 1L)
  obj$set_index(1L)
  y0 <- obj$state()
  y22_1 <- obj$run(5)
  y22_2 <- obj$state()

  ## And again without parallel
  obj <- res$new(list(sd = 1), 0, 10, n_threads = 1L, seed = 1L)
  obj$set_index(1L)
  y12_1 <- obj$run(5)
  y12_2 <- obj$state()

  ## Quick easy check:
  expect_equal(y22_1[1, ], y22_2[1, ])
  expect_equal(y12_1[1, ], y12_2[1, ])

  if (obj$has_openmp() && y0[2, 1] == 1) {
    ## OMP is enabled (note: this relies on implementation details of
    ## openmp and we'd need to change this for a CRAN release - what
    ## we'd expect to see is that 0 and 1 are both present, but as
    ## we're leaving it up to the compiler to set the schedule I
    ## believe this is not reliable theoretically, even though it is
    ## empirically)
    expect_equal(y22_2[2, ], rep(0:1, each = 5))
  } else {
    expect_equal(y22_2[2, ], rep(-1, 10))
  }

  expect_equal(y22_1, y12_1)
})


## It's quite hard to test the cache well; we might wrap it up a bit
## nicer than it is. The trick is we can't safely unload things
## (because the finalisers might run) and we can't easily purge the
## cache and show that it *would* compile either. And we can't just
## compile and load the same thing twice because the names of the dlls
## will clash.
##
## This at least shows that on the second time round we don't compile
## and we get the right data out.
test_that("cache hits do not compile", {
  cmp <- dust(dust_file("examples/walk.cpp"), quiet = TRUE)

  mock <- mockery::mock()
  res <- with_mock(
    "pkgbuild::compile_dll" = mock,
    dust(dust_file("examples/walk.cpp"), quiet = TRUE))
  ## Never called
  expect_equal(mockery::mock_calls(mock), list())
  ## Same object
  expect_identical(res, cmp)
})


test_that("set model state and time, varying time", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(rep(as.numeric(1:2), each = 10), 10, 2)
  step <- 0:1
  mod$set_state(m, step)
  cmp <- dust_rng$new(1, 1)$rnorm(10, 0, 1)

  state <- mod$state()
  expect_equal(mod$step(), 1)
  expect_equal(state[, 2], m[, 2])
  expect_equal(state[, 1], m[, 1] + cmp)
})


test_that("setting model state and step requires correct length step", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 5, seed = 1L)
  m <- matrix(rep(as.numeric(1:2), each = 10), 10, 5)
  expect_error(
    mod$set_state(m, 0:3),
    "Expected 'step' to be scalar or length 5")
  expect_error(
    mod$set_state(m, 0:7),
    "Expected 'step' to be scalar or length 5")
})


test_that("set model state and time, constant time", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(runif(20), 10, 2)
  step <- 10L
  mod$set_state(m, step)

  state <- mod$state()
  expect_equal(mod$step(), 10)
  expect_equal(state, m)
})


test_that("set model time but not state", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  expect_null(mod$set_state(NULL, NULL))
  expect_equal(mod$step(), 0)
  expect_equal(mod$state(), matrix(1:10, 10, 2))

  expect_null(mod$set_state(NULL, 10L))
  expect_equal(mod$step(), 10)
  expect_equal(mod$state(), matrix(1:10, 10, 2))
})


test_that("NULL state leaves state untouched", {
  res <- dust_example("variable")
  mod <- res$new(list(len = 10), 0, 2, seed = 1L)
  m <- matrix(runif(20), 10, 2)
  mod$set_state(m, NULL)
  expect_equal(mod$state(), m)
  expect_equal(mod$step(), 0)

  mod$set_state(NULL, 10L)
  expect_equal(mod$state(), m)
  expect_equal(mod$step(), 10)

  mod$set_state(NULL, NULL)
  expect_equal(mod$state(), m)
  expect_equal(mod$step(), 10)
})


test_that("type coersion in setting index", {
  res <- dust_example("sir")
  obj <- res$new(list(), 0, 100, seed = 1L)
  expect_null(obj$set_index(1L))
  expect_null(obj$set_index(1))
  expect_error(
    obj$set_index(1.5),
    "All elements of 'index' must be integer-like",
    fixed = TRUE)
  expect_error(
    obj$set_index(TRUE),
    "Expected a numeric vector for 'index'",
    fixed = TRUE)
})


test_that("can't load invalid example", {
  expect_error(dust_example("asdf"), "Unknown example 'asdf'")
})


test_that("can run volatility example", {
  res <- dust_example("volatility")
  obj <- res$new(list(), 0, 5000, seed = 1L)
  y <- drop(dust_iterate(obj, 0:100))
  expect_lt(diff(range(colMeans(y))), 0.5)
})


test_that("has_openmp can be called statically and normally", {
  res <- dust_example("volatility")
  expected <- openmp_info()$has_openmp
  expect_equal(res$public_methods$has_openmp(), expected)
  expect_equal(res$new(list(), 0, 1)$has_openmp(), expected)
})
