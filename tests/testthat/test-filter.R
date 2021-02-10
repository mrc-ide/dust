context("filter")

test_that("Can run the filter", {
  dat <- example_filter()

  np <- 10
  mod <- dat$model$new(list(), 0, np, seed = 10L)
  expect_error(mod$filter(), "Data has not been set for this object")

  mod$set_data(dat$dat_dust)

  ## We can perform the entire particle filter manually with the C
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
  cmp_history <- filter_history_reorder(hv, hi)

  mod <- dat$model$new(list(), 0, np, seed = 10L)
  mod$set_data(dat$dat_dust)
  ans <- mod$filter()
  expect_equal(ans,
               list(log_likelihood = cmp_log_likelihood,
                    history = NULL))

  mod <- dat$model$new(list(), 0, np, seed = 10L)
  mod$set_data(dat$dat_dust)
  ans <- mod$filter(TRUE)
  expect_equal(ans,
               list(log_likelihood = cmp_log_likelihood,
                    history = cmp_history))
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
    dat$model$new(pars[[2]], 0, np, seed = seed[-(1:320)]))
  for (i in seq_along(cmp)) {
    cmp[[i]]$set_data(dat$dat_dust)
  }
  cmp_res <- lapply(cmp, function(el) el$filter(TRUE))

  expect_length(ans$log_likelihood, 2)
  expect_equal(ans$log_likelihood[[1]], cmp_res[[1]]$log_likelihood)
  expect_equal(ans$log_likelihood[[2]], cmp_res[[2]]$log_likelihood)

  expect_equal(dim(ans$history), c(5, 10, 2, 151))
  expect_equal(ans$history[, , 1, ], cmp_res[[1]]$history)
  expect_equal(ans$history[, , 2, ], cmp_res[[2]]$history)
})


test_that("can filter history using index", {
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
  expect_equal(ans1$history, ans2$history[c(1, 5), , , ])
})