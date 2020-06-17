context("utils")

test_that("null-or-value works", {
  expect_equal(1 %||% NULL, 1)
  expect_equal(1 %||% 2, 1)
  expect_equal(NULL %||% NULL, NULL)
  expect_equal(NULL %||% 2, 2)
})


test_that("valid name", {
  x <- "1name"
  expect_error(assert_valid_name(x),
               "'x' must contain only letters and numbers")
  expect_error(assert_valid_name("foo_bar"),
               "'.+' must contain only letters and numbers")
  expect_error(assert_valid_name("foo.bar"),
               "'.+' must contain only letters and numbers")
})
