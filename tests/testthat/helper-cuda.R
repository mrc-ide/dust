example_cuda_config <- function() {
  list(has_cuda = TRUE,
       cuda_version = numeric_version("11.1.243"),
       devices = data.frame(
         id = 0L, name = "GeForce RTX 2080 Ti", memory = 11016.3125,
         version = 75L, stringsAsFactors = FALSE))
}

mock_create_test_package <- function(...) {
  path <- tempfile()
  suffix <- paste(sample(c(0:9, letters[1:6]), 8, TRUE), collapse = "")
  base <- paste0("dust", suffix)
  dir.create(file.path(path, "R"), FALSE, TRUE)
  writeLines(sprintf("Package: %s\nVersion: 0.1", base),
             file.path(path, "DESCRIPTION"))
  file.create(file.path(path, "NAMESPACE"))
  code <- sprintf("dust_gpu_info <- function() %s",
                  paste(deparse(example_cuda_config()), collapse = "\n"))
  writeLines(code, file.path(path, "R", "code.R"))
  list(path = path, name = base)
}


create_launch_control <- function(block_size, block_count) {
  list(block_size = block_size, block_count = block_count,
       shared_size_bytes = 0, shared_int = FALSE, shared_real = FALSE)
}
