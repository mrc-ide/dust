`%||%` <- function(a, b) { # nolint
  if (is.null(a)) b else a
}


normalize_path <- function(path) {
  normalizePath(path, "/", mustWork = TRUE)
}


hash_file <- function(path, short = TRUE) {
  hash <- unname(tools::md5sum(path))
  if (short) {
    hash <- substr(hash, 1, 8)
  }
  hash
}


dust_file <- function(path) {
  system.file(path, package = "dust", mustWork = TRUE)
}


read_lines <- function(path) {
  paste(readLines(path), collapse = "\n")
}
