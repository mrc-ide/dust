#!/usr/bin/env Rscript
dest_include <- "inst/include/trng"
dest_src <- "src/trng"
dest_licence <- "inst/COPYING.trng"
unlink(dir(dest_include, full.names = TRUE))
unlink(dir(dest_src, full.names = TRUE))
unlink(dest_licence)

file_copy <- function (..., overwrite = TRUE) {
  ok <- file.copy(..., overwrite = overwrite)
  if (any(!ok)) {
    stop("Error copying files")
  }
  ok
}

version <- "4.23"
url <- sprintf("https://github.com/rabauke/trng4/archive/v%s.tar.gz", version)
zip <- tempfile()
download.file(url, zip, mode = "wb")

tmp <- tempfile()
untar(zip, exdir = tmp)
path <- dir(tmp, full.names = TRUE)
path_src <- file.path(path, "trng")

dir.create(dest_include, FALSE, TRUE)
dir.create(dest_src, FALSE, TRUE)

file_copy(file.path(path, "COPYING"), dest_licence)
file_copy(list.files(path_src, pattern = "\\.hpp$", full.names = TRUE),
          dest_include)

for (s in list.files(path_src, pattern = "\\.cc$", full.names = TRUE)) {
  txt <- readLines(s)
  writeLines(sub('#include "(.*\\.hpp)"', "#include <trng/\\1>", txt),
             file.path(dest_src, basename(s)))
}
