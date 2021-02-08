context("data")

test_that("Can construct dust_data", {
  d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  res <- dust_data(d)
  expect_is(res, "list")
  expect_null(names(res))
  expect_length(res, 6)
  expect_equal(lengths(res), rep(2, 6))
  expect_identical(lapply(res, "[[", 1), list(0L, 10L, 20L, 30L, 40L, 50L))
  expect_identical(res[[4]][[2]], as.list(d[4, ]))
})


test_that("Can validate step", {
  d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  expect_error(
    dust_data(d, "col"),
    "'col' is not a column in d")

  d$step[[4]] <- -40
  expect_error(
    dust_data(d),
    "All elements in column 'step' must be nonnegative")

  d$step[[4]] <- 40.1
  expect_error(
    dust_data(d),
    "All elements in column 'step' must be integer-like")

  d$step[[4]] <- 50
  expect_error(
    dust_data(d),
    "All elements in column 'step' must be unique")
})


test_that("rounding errors are converted to integers", {
  d <- data.frame(t = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  d$t <- d$t + sample(c(-1e-14, 1e-14), 6, replace = TRUE)
  res <- dust_data(d, "t")
  expect_identical(lapply(res, "[[", 1), list(0L, 10L, 20L, 30L, 40L, 50L))
})


test_that("multiple data, shared", {
  d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
  res <- dust_data(d, multi = 3L)

  expect_is(res, "list")
  expect_null(names(res))
  expect_length(res, 6)
  expect_equal(lengths(res), rep(4, 6))
  expect_identical(lapply(res, "[[", 1), list(0L, 10L, 20L, 30L, 40L, 50L))
  expect_identical(res[[4]][[2]], as.list(d[4, ]))
})


test_that("multiple data, different", {
  d <- data.frame(step = rep(seq(0, 50, by = 10), 3),
                  group = factor(rep(c("a", "b", "c"), each = 6)),
                  a = runif(18), b = runif(18))

  res <- dust_data(d, multi = "group")
  expect_is(res, "list")
  expect_length(res, 6)
  expect_equal(lengths(res), rep(4, 6))
  expect_identical(lapply(res, "[[", 1), list(0L, 10L, 20L, 30L, 40L, 50L))
  expect_identical(res[[4]][[2]], as.list(d[4, ]))
  expect_identical(res[[4]][[3]], as.list(d[10, ]))
  expect_identical(res[[4]][[4]], as.list(d[16, ]))

  ## Order of the grouping variable is not important so long as steps
  ## are consistent:
  expect_identical(
    dust_data(d[order(d$group, decreasing = TRUE), ], multi = "group"),
    res)
  expect_identical(
    dust_data(d[order(d$step, d$group), ], multi = "group"),
    res)
})


test_that("validate multiple data", {
  d <- data.frame(step = rep(seq(0, 50, by = 10), 3),
                  group = factor(rep(c("a", "b", "c"), each = 6)),
                  a = runif(18), b = runif(18))
  expect_error(
    dust_data(d, multi = "grp"),
    "'grp' is not a column in d")
  expect_error(
    dust_data(d, multi = "a"),
    "Column 'a' must be a factor")
  expect_error(
    dust_data(cbind(d, grp = as.character(d$group), stringsAsFactors = FALSE),
              multi = "grp"),
    "Column 'grp' must be a factor")
  expect_error(
    dust_data(d[-14, ], multi = "group"),
    "All groups must have the same time steps, in the same order")
  expect_error(
    dust_data(d[sample(nrow(d)), ], multi = "group"),
    "All groups must have the same time steps, in the same order")
  expect_error(
    dust_data(d, multi = TRUE),
    "Invalid option for 'multi'; must be NULL, integer or character")
})
