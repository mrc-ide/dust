context("filter")

test_that("Can run the filter", {
  dat <- example_filter()

  np <- 10
  mod <- dat$model$new(list(), 0, np, seed = 10L)
  expect_error(mod$filter(), "Data has not been set for this object")

  mod$set_data(dat$dat_dust)

  ## We can perform the entire particle filter manually with the C++
  ## version, and this will run entirely on the dust generator
  n_data <- length(dat$dat_dust)
  ll <- numeric(n_data)
  hv <- array(NA_real_, c(5L, np, n_data + 1L))
  hi <- matrix(NA_integer_, np, n_data + 1L)
  hv[, , 1] <- mod$state()
  hi[, 1] <- seq_len(np)
  for (i in seq_len(n_data)) {
    mod$run(dat$dat_dust[[i]][[1]])
    hv[, , i + 1] <- mod$state()
    weights <- mod$compare_data()
    tmp <- scale_log_weights(weights)
    ll[[i]] <- tmp$average
    idx <- mod$resample(tmp$weights)
    hi[, i + 1] <- idx
  }
  cmp_log_likelihood <- Reduce(`+`, ll) # naive sum()
  cmp_trajectories <- filter_trajectories_reorder(hv, hi)

  mod <- dat$model$new(list(), 0, np, seed = 10L)
  mod$set_data(dat$dat_dust)
  ans <- mod$filter()
  expect_equal(ans,
               list(log_likelihood = cmp_log_likelihood,
                    trajectories = NULL,
                    snapshots = NULL))

  mod <- dat$model$new(list(), 0, np, seed = 10L)
  mod$set_data(dat$dat_dust)
  ans <- mod$filter(TRUE)
  expect_equal(ans,
               list(log_likelihood = cmp_log_likelihood,
                    trajectories = cmp_trajectories,
                    snapshots = NULL))
})


test_that("Can run multiple filters at once", {
  dat <- example_filter()

  np <- 10
  pars <- list(list(beta = 0.2), list(beta = 0.1))
  mod <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  seed <- mod$rng_state()

  expect_error(mod$filter(), "Data has not been set for this object")
  mod$set_data(dust_data(dat$dat, multi = 2))
  ans <- mod$filter(TRUE)

  cmp <- list(
    dat$model$new(pars[[1]], 0, np, seed = seed[1:320]),
    dat$model$new(pars[[2]], 0, np, seed = seed[321:640]))
  for (i in seq_along(cmp)) {
    cmp[[i]]$set_data(dat$dat_dust)
  }
  cmp_res <- lapply(cmp, function(el) el$filter(TRUE))

  expect_length(ans$log_likelihood, 2)
  expect_equal(ans$log_likelihood[[1]], cmp_res[[1]]$log_likelihood)
  expect_equal(ans$log_likelihood[[2]], cmp_res[[2]]$log_likelihood)

  expect_equal(dim(ans$trajectories), c(5, 10, 2, 151))
  expect_equal(ans$trajectories[, , 1, ], cmp_res[[1]]$trajectories)
  expect_equal(ans$trajectories[, , 2, ], cmp_res[[2]]$trajectories)
})


test_that("can filter trajectories using index", {
  dat <- example_filter()

  np <- 10
  pars <- list(list(beta = 0.2), list(beta = 0.1))

  mod1 <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod1$set_index(c(1L, 5L))
  mod1$set_data(dust_data(dat$dat, multi = 2))
  ans1 <- mod1$filter(TRUE)

  mod2 <- dat$model$new(pars, 0, np, seed = 10L, pars_multi = TRUE)
  mod2$set_data(dust_data(dat$dat, multi = 2))
  ans2 <- mod2$filter(TRUE)

  expect_equal(ans1$log_likelihood, ans2$log_likelihood)
  expect_equal(ans1$trajectories, ans2$trajectories[c(1, 5), , , ])
})


test_that("log weights scaling calculation is correct", {
  w <- runif(10)
  lw <- log(w)
  res <- cpp_scale_log_weights(lw)

  expect_equal(max(res$weights), 1)
  expect_equal(res$weights, w / max(w))
  expect_equal(exp(res$average), mean(w))
})


test_that("scale log weights copes with NaN", {
  w <- log(runif(10))
  expect_equal(scale_log_weights(w),
               cpp_scale_log_weights(w))

  w[3] <- NaN
  res <- cpp_scale_log_weights(w)
  expect_equal(res$weights[[3]], 0)
  expect_equal(res, scale_log_weights(w))

  w[3] <- NA
  res <- cpp_scale_log_weights(w)
  expect_equal(res$weights[[3]], 0)
  expect_equal(res, scale_log_weights(w))

  w[3] <- -Inf
  res <- cpp_scale_log_weights(w)
  expect_equal(res$weights[[3]], 0)
  expect_equal(res, scale_log_weights(w))
})


test_that("Can save out state during a run", {
  dat <- example_filter()

  np <- 10
  mod <- dat$model$new(list(), 0, np, seed = 10L)
  mod$set_data(dat$dat_dust)

  ## TODO: handle better doubles here using as_integer
  step_snapshot <- c(100L, 200L, 400L)

  ## We can perform the entire particle filter manually with the C
  ## version, and this will run entirely on the dust generator
  s <- array(NA_real_, c(5L, np, length(step_snapshot)))

  for (i in seq_along(dat$dat_dust)) {
    to <- dat$dat_dust[[i]][[1]]
    mod$run(to)
    weights <- mod$compare_data()
    tmp <- scale_log_weights(weights)
    idx <- mod$resample(tmp$weights)

    j <- match(to, step_snapshot)
    if (!is.na(j)) {
      s[, , j] <- mod$state()
    }
  }

  mod <- dat$model$new(list(), 0, np, seed = 10L)
  mod$set_data(dat$dat_dust)
  ans <- mod$filter(step_snapshot = step_snapshot)
  expect_equal(ans$snapshots, s)
})


test_that("Validate step_snapshot", {
  dat <- example_filter()
  mod <- dat$model$new(list(), 0, 10, seed = 10L)

  mod$set_data(dat$dat_dust)
  expect_error(
    mod$filter(step_snapshot = c(100.1, 200.1, 400.1)),
    "All elements of 'step_snapshot' must be integer-like")
  expect_error(
    mod$filter(step_snapshot = c(100, -200, 400)),
    "'step_snapshot' must be positive")
  expect_error(
    mod$filter(step_snapshot = c(100, 400, 200)),
    "'step_snapshot' must be strictly increasing")
  expect_error(
    mod$filter(step_snapshot = c(100, 200, 200)),
    "'step_snapshot' must be strictly increasing")
  expect_error(
    mod$filter(step_snapshot = c(100, 201, 400)),
    "'step_snapshot[2]' (step 201) was not found in data",
    fixed = TRUE)
})
