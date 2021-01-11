context("multi")

test_that("create trivial multi dust object", {
  res <- dust_example("walk")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, data_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1)), 0, 10, seed = 1L, data_multi = TRUE)

  expect_identical(obj2$name(), obj1$name())
  expect_identical(obj2$param(), obj1$param())
  expect_identical(obj2$n_threads(), obj1$n_threads())
  expect_identical(obj2$has_openmp(), obj1$has_openmp())
  expect_identical(obj2$step(), obj1$step())

  expect_equal(obj2$data(), list(obj1$data()))
  expect_equal(obj2$info(), list(obj1$info()))

  expect_identical(obj2$rng_state(), obj1$rng_state())

  expect_identical(obj1$n_data(), 0L)
  expect_identical(obj2$n_data(), 1L)

  expect_identical(obj2$state(), array(obj1$state(), c(1, 10, 1)))
  expect_identical(obj2$state(1L), array(obj1$state(1L), c(1, 10, 1)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
})


test_that("create trivial 2 element mulitdust object", {
  res <- dust_example("walk")
  obj1 <- res$new(list(sd = 1), 0, 10, seed = 1L, data_multi = FALSE)
  obj2 <- res$new(list(list(sd = 1), list(sd = 1)), 0, 5, seed = 1L,
                  data_multi = TRUE)

  expect_identical(obj2$n_data(), 2L)
  expect_equal(obj2$state(), array(obj1$state(), c(1, 5, 2)))

  y1 <- obj1$run(1)
  y2 <- obj2$run(1)
  expect_equal(y2, array(y1, c(1, 5, 2)))

  expect_identical(obj1$rng_state(), obj2$rng_state())
})


test_that("Can particles and resume continues with rng", {
  res <- dust_example("walk")
  sd1 <- 2
  sd2 <- 4
  sd3 <- 8

  data1 <- list(list(sd = sd1), list(sd = sd2))
  data2 <- list(list(sd = sd2), list(sd = sd3))

  nd <- length(data1)
  np <- 10
  obj <- res$new(data1, 0, np, seed = 1L, data_multi = TRUE)

  ns <- 5
  y1 <- obj$run(ns)
  expect_equal(obj$step(), ns)
  obj$reset(data2, 0)
  expect_equal(obj$step(), 0)
  y2 <- obj$run(ns)
  expect_equal(obj$step(), ns)

  ## Then draw the random numbers:
  rng <- dust_rng$new(1, 20)
  m1 <- array(rng$rnorm(ns * nd * np, 0, 1), c(np, nd, ns))
  m2 <- array(rng$rnorm(ns * nd * np, 0, 1), c(np, nd, ns))

  expect_equal(
    array(y1, c(np, nd)),
    apply(m1, 1:2, sum) * rep(c(sd1, sd2), each = np))

  expect_equal(
    array(y2, c(np, nd)),
    apply(m2, 1:2, sum) * rep(c(sd2, sd3), each = np))
})


test_that("Can set state", {
  res <- dust_example("variable")

  nd <- 3
  ns <- 7
  np <- 13

  data <- rep(list(list(len = ns)), nd)
  mod <- res$new(data, 0, np, seed = 1L, data_multi = TRUE)

  ## Initial state:
  expect_equal(
    mod$state(),
    array(1:ns, c(ns, np, nd)))

  y <- mod$state()
  y[] <- runif(length(y))

  mod$set_state(y)
  expect_equal(mod$state(), y)

  expect_error(
    mod$set_state(c(y)),
    "Expected a 3d array for 'state' (but recieved a vector)",
    fixed = TRUE)
  expect_error(
    mod$set_state(matrix(c(y), c(ns, nd * np))),
    "Expected a 3d array for 'state'")
  expect_error(
    mod$set_state(y[-1, , ]),
    "Expected a 3d array with 7 rows for 'state'")
  expect_error(
    mod$set_state(y[, -1, ]),
    "Expected a 3d array with 13 columns for 'state'")
  expect_error(
    mod$set_state(y[, , -1]),
    "Expected a 3d array with dim[3] == 3 for 'state'",
    fixed = TRUE)

  ## Unchanged
  expect_equal(mod$state(), y)
})


test_that("can set index", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  data <- rep(list(list(len = ns)), nd)
  mod <- res$new(data, 0, np, seed = 1L, data_multi = TRUE)
  y <- mod$state()
  y[] <- runif(length(y))
  mod$set_state(y)
  idx <- c(1, 3, 5)
  mod$set_index(idx)

  expect_equal(mod$state(), y)
  expect_equal(mod$state(idx), y[idx, , ])
  expect_equal(mod$run(0), y[idx, , ])
})


test_that("can error if out-of-bounds index used", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  data <- rep(list(list(len = ns)), nd)
  mod <- res$new(data, 0, np, seed = 1L, data_multi = TRUE)
  expect_error(mod$set_index(c(1, 9, 2)),
               "All elements of 'index' must lie in [1, 7]",
               fixed = TRUE)
})


test_that("Can reorder particles", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  data <- rep(list(list(len = ns)), nd)
  mod <- res$new(data, 0, np, seed = 1L, data_multi = TRUE)
  y <- array(as.numeric(seq_len(nd * ns * np)), c(ns, np, nd))
  mod$set_state(y)

  ## Our reorder matrix:
  i <- replicate(nd, sample.int(np, np, replace = TRUE))

  ## Compupting the reorder is actually hard
  cmp <- array(c(y[, i[, 1], 1],
                 y[, i[, 2], 2],
                 y[, i[, 3], 3]), dim(y))
  mod$reorder(i)
  expect_equal(mod$state(), cmp)
})


test_that("Can reorder particles", {
  res <- dust_example("variable")
  nd <- 3
  ns <- 7
  np <- 13
  data <- rep(list(list(len = ns)), nd)
  mod <- res$new(data, 0, np, seed = 1L, data_multi = TRUE)
  y <- array(as.numeric(seq_len(nd * ns * np)), c(ns, np, nd))
  mod$set_state(y)

  ## Our reorder matrix:
  i <- replicate(nd, sample.int(np, np, replace = TRUE))
  expect_error(
    mod$reorder(c(i)),
    "Expected a matrix for 'index'")
  expect_error(
    mod$reorder(i[-1, ]),
    "Expected a matrix with 13 rows for 'index'")
  expect_error(
    mod$reorder(i[, -1]),
    "Expected a matrix with 3 columns for 'index'")
  expect_equal(mod$state(), y)
})
