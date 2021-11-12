test_that("xoshiro output agrees with reference data", {
  path <- "xoshiro-ref"
  status <- list()
  for (name in dir(path)) {
    obj <- dust_rng_pointer$new(seed = 42, algorithm = name)
    res <- test_xoshiro_run(obj)
    obj$sync()
    len <- if (grepl("^xoshiro128", name)) 4 else 8
    s <- matrix(obj$state(), len)[rev(seq_len(len)), ]
    s_str <- apply(s, 2, paste, collapse = "")
    cmp <- readLines(sprintf("%s/%s", path, name))
    expect_equal(c(res, s_str), cmp)
  }
})
