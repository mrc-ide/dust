context("metadata")

test_that("parse sir model metadata", {
  meta <- parse_metadata(dust_file("examples/sir.cpp"))
  expect_equal(meta$type, "sir")
  expect_equal(meta$name, "sir")
  expect_equal(
    meta$param,
    list(beta = list(required = FALSE),
         gamma = list(required = FALSE)))
})
