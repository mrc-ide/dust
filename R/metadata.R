parse_metadata <- function(filename) {
  data <- decor::cpp_decorations(files = filename)
  ret <- list(name = parse_metadata_name(data),
              class = parse_metadata_class(data),
              param = parse_metadata_param(data),
              time_type = parse_metadata_time_type(data),
              has_gpu_support = parse_metadata_has_gpu_support(data))

  if (is.null(ret$class)) {
    ret$class <- parse_metadata_guess_class(readLines(filename))
  }

  if (is.null(ret$name)) {
    ret$name <- ret$class
  }

  if (is.null(ret$has_gpu_support)) {
    ret$has_gpu_support <- parse_code_has_gpu_support(readLines(filename))
  }

  ret
}


parse_metadata_simple <- function(data, attribute) {
  i <- data$decoration == attribute
  if (!any(i)) {
    return(NULL)
  }
  if (sum(i) > 1) {
    stop(sprintf("More than one [[%s()]] attribute found %s",
                 attribute, parse_metadata_describe(data, i)))
  }
  value <- data$params[[which(i)]]
  if (length(value) != 1L || value[[1]] == attribute) {
    stop(sprintf("Expected [[%s()]] to have one argument %s",
                 attribute, parse_metadata_describe(data, i)))
  }
  if (any(nzchar(names(value)))) {
    stop(sprintf("Argument to [[%s()]] attribute must be unnamed %s",
                 attribute, parse_metadata_describe(data, i)))
  }
  as.character(value[[1]])
}


parse_metadata_name <- function(data) {
  name <- parse_metadata_simple(data, "dust::name")
  if (!is.null(name) && !grepl("^[A-Za-z][A-Za-z0-9_]*$", name)) {
    stop(paste("'[[dust::name]]' must contain only letters, numbers and",
               "underscores, starting with a letter"))
  }
  name
}


parse_metadata_class <- function(data) {
  parse_metadata_simple(data, "dust::class")
}


parse_metadata_time_type <- function(data) {
  value <- parse_metadata_simple(data, "dust::time_type") %||% "discrete"
  if (!is.null(value)) {
    valid <- c("continuous", "discrete")
    if (!(value %in% valid)) {
      stop(sprintf("Invalid value for dust::time_type, expected one of %s",
                   paste(squote(valid), collapse = ", ")))
    }
  }
  value
}


parse_metadata_has_gpu_support <- function(data) {
  value <- parse_metadata_simple(data, "dust::has_gpu_support")
  if (!is.null(value)) {
    if (!grepl("^(true|false)$", value, ignore.case = TRUE)) {
      stop("Invalid value for dust::has_gpu_support, expected logical")
    }
    value <- as.logical(toupper(value))
  }
  value
}


parse_metadata_param <- function(data) {
  i <- data$decoration == "dust::param"
  if (!any(i)) {
    return(NULL)
  }
  value <- lapply(which(i), parse_metadata_param1, data)

  nms <- vcapply(value, "[[", "name")
  if (any(duplicated(nms))) {
    dups <- nms[duplicated(nms)]
    stop(sprintf(
      "Duplicated [[dust::param()]] attributes: %s %s",
      paste(squote(unique(dups)), collapse = ", "),
      parse_metadata_describe(data, which(i)[nms %in% dups])))
  }

  set_names(lapply(value, "[[", "data"), nms)
}


parse_metadata_param1 <- function(i, data) {
  x <- data$params[[i]]
  if (length(x) == 0) {
    stop(sprintf("At least one argument required to [[dust::param()]] %s",
                 parse_metadata_describe(data, i)),
         call. = FALSE)

  }
  if (nzchar(names(x)[[1]])) {
    stop(sprintf("First argument of [[dust::param()]] must be unnamed %s",
                 parse_metadata_describe(data, i)),
         call. = FALSE)
  }
  if (any(!nzchar(names(x)[-1]))) {
    stop(sprintf(
      "Arguments 2 and following of of [[dust::param]] must be named %s",
      parse_metadata_describe(data, i)),
      call. = FALSE)
  }

  list(name = as.character(x[[1]]),
       data = lapply(x[-1], function(el)
         if (is.symbol(el)) as.character(el) else el))
}


parse_metadata_describe <- function(data, i) {
  err <- data[i, ]
  if (nrow(err) == 1L) {
    sprintf("%s:%s", basename(err$file), err$line)
  } else {
    sprintf("%s:(%s)", basename(err$file[[1]]),
            paste(err$line, collapse = ", "))
  }
}


parse_metadata_guess_class <- function(txt) {
  re <- "^\\s*class\\s+([^{ ]+)\\s*(\\{.*|$)"
  i <- grep(re, txt)
  if (length(i) != 1L) {
    stop("Could not automatically detect class name; add [[dust::class()]]?")
  }
  sub(re, "\\1", txt[[i]])
}


parse_code_has_gpu_support <- function(txt) {
  re <- "void\\s+update_gpu\\s*<\\s*"
  any(grepl(re, txt))
}


deparse_param <- function(x) {
  n <- length(x)
  if (n == 0L) {
    return("NULL")
  }
  str <- vcapply(x, function(x) paste(deparse(x, 120L), collapse = "\n"))
  start <- rep(c("list(", "     "), c(1L, n - 1L))
  end <- rep(c(",", ")"), c(n - 1L, 1L))
  paste(sprintf("%s%s = %s%s", start, names(x), str, end), collapse = "\n")
}
