`%||%` <- function(a, b) { # nolint
  if (is.null(a)) b else a
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


assert_valid_name <- function(x, name = deparse(substitute(x))) {
  if (!grepl("^[A-Za-z][A-Zxa-z0-9]*$", x)) {
    stop(sprintf(
      "'%s' must contain only letters and numbers, starting with a letter",
      name))
  }
}


assert_file_exists <- function(path, name = "File") {
  if (!file.exists(path)) {
    stop(sprintf("%s '%s' does not exist", name, path))
  }
}


is_directory <- function(path) {
  file.info(path)$isdir
}
