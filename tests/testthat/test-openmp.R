test_that("dust_openmp_info contains expected fields", {
  info <- dust_openmp_support()
  expect_type(info, "list")
  nms <- c("num_procs", "max_threads", "thread_limit", "openmp_version",
           "has_openmp", "mc.cores", "OMP_THREAD_LIMIT", "OMP_NUM_THREADS",
           "MC_CORES", "limit_r", "limit_openmp", "limit")
  expect_equal(names(info), nms)
})


test_that("dust_openmp_info contains expected fields", {
  skip_on_cran()
  info1 <- dust_openmp_support()
  info2 <- dust_openmp_support(TRUE)
  expect_equal(
    setdiff(names(info2), names(info1)),
    "has_openmp_compiler")
  expect_equal(
    setdiff(names(info1), names(info2)),
    character())
  expect_equal(info2$has_openmp_compiler, cache$has_openmp_compiler)
})


test_that("check limit", {
  expect_error(
    openmp_check_limit(10, 2, "error"),
    "Requested number of threads '10' exceeds a limit of '2'")

  expect_message(
    res <- openmp_check_limit(10, 2, "message"),
    "Requested number of threads '10' exceeds a limit of '2'")
  expect_equal(
    res, 10)

  expect_message(
    res <- openmp_check_limit(10, 2, "fix"),
    "Requested number of threads '10' exceeds a limit of '2'")
  expect_equal(
    res, 2)

  expect_error(
    openmp_check_limit(10, 2, "increase"),
    "'action' must be one of 'error', 'message', 'fix'")
})


test_that("limit is 1 if openmp not supported", {
  skip_if_not_installed("mockery")
  mock_info <- mockery::mock(
    list(num_procs = NA_integer_, max_threads = NA_integer_,
         thread_limit = NA_integer_,  openmp_version = NA_integer_,
         has_openmp = FALSE))
  mockery::stub(openmp_info, "cpp_openmp_info", mock_info)
  res <- openmp_info()

  mockery::expect_called(mock_info, 1L)
  expect_equal(res$limit, 1)
  expect_identical(res$limit_openmp, 1L)
})


test_that("limit is more than 1 if openmp supported", {
  skip_if_not_installed("mockery")
  mock_info <- mockery::mock(
    list(num_procs = 8L, max_threads = 8L,
         thread_limit = 1024L, openmp_version = 201511L,
         has_openmp = TRUE))
  mockery::stub(openmp_info, "cpp_openmp_info", mock_info)
  res <- withr::with_options(
    list(mc.cores = 4),
    openmp_info())
  mockery::expect_called(mock_info, 1L)
  expect_equal(res$limit, 4)
  expect_equal(res$limit_openmp, 8L)
  expect_equal(res$limit_r, 4)
})


test_that("order of preference for R's limits", {
  expect_equal(
    withr::with_options(
      list(mc.cores = 4),
      withr::with_envvar(
        c("MC_CORES" = 8),
        unname(openmp_info()[c("mc.cores", "MC_CORES", "limit_r")]))),
    list(4, 8, 4))

  expect_equal(
    withr::with_options(
      list(mc.cores = 8),
      withr::with_envvar(
        c("MC_CORES" = 4),
        unname(openmp_info()[c("mc.cores", "MC_CORES", "limit_r")]))),
    list(8, 4, 8))
})


test_that("detect compilation failure", {
  skip_if_not_installed("mockery")
  mock_dust <- mockery::mock(
    stop("compilation failed!"))
  mockery::stub(has_openmp_compiler_test, "dust", mock_dust)
  expect_false(has_openmp_compiler_test())
})


test_that("detect compilation success, no support", {
  skip_if_not_installed("mockery")
  mock_state <- mockery::mock(-1)
  mock_dust <- mockery::mock(
    R6::R6Class(public = list(
                  initialize = function(...) NULL,
                  run = function(...) NULL,
                  state = function(...) mock_state(...))))
  mockery::stub(has_openmp_compiler_test, "dust", mock_dust)
  expect_false(has_openmp_compiler_test())
  mockery::expect_called(mock_state, 1L)
  expect_equal(mockery::mock_args(mock_state)[[1]], list(2L))
})


test_that("detect compilation success, with support", {
  skip_if_not_installed("mockery")
  mock_state <- mockery::mock(0)
  mock_dust <- mockery::mock(
    R6::R6Class(public = list(
                  initialize = function(...) NULL,
                  run = function(...) NULL,
                  state = function(...) mock_state(...))))
  mockery::stub(has_openmp_compiler_test, "dust", mock_dust)
  expect_true(has_openmp_compiler_test())
  mockery::expect_called(mock_state, 1L)
  expect_equal(mockery::mock_args(mock_state)[[1]], list(2L))
})


test_that("dust_openmp_threads interface works", {
  expect_equal(
    dust_openmp_threads(NULL),
    dust_openmp_support()$limit)
  expect_equal(
    suppressMessages(dust_openmp_threads(1000, action = "message")),
    1000)
  expect_equal(
    suppressMessages(dust_openmp_threads(1000, action = "fix")),
    dust_openmp_support()$limit)
  expect_equal(
    dust_openmp_threads(1, action = "error"),
    1)
})
