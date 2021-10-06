test_that("xoshiro output agrees with reference data", {
  path <- "xoshiro-ref"
  status <- list()
  for (name in dir(path)) {
    res <- test_xoshiro_run(name)
    cmp <- readLines(sprintf("%s/%s", path, name))
    expect_equal(res, cmp)
  }
})
