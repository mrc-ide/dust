parse_metadata <- function(filename) {
  data <- decor::cpp_decorations(files = filename)
  ret <- list(name = parse_metadata_name(data),
              type = parse_metadata_type(data),
              param = parse_metadata_param(data))

  if (is.null(ret$type)) {
    ret$type <- parse_metadata_guess_type(readLines(filename))
  }

  if (is.null(ret$name)) {
    ret$name <- ret$type
  }

  ret
}

parse_metadata_name <- function(data) {
  i <- data$decoration == "dust::name"
  if (!any(i)) {
    return(NULL)
  }
  if (sum(i) > 1) {
    stop("More than one dust::name decoration found")
  }
  value <- data$params[[which(i)]]
  if (length(value) != 1L) {
    stop("Expected dust::name to have one argument")
  }
  if (any(nzchar(names(value)))) {
    stop("Invalid format for [[dust::name()]] attribute")
  }
  as.character(value[[1]])
}


parse_metadata_type <- function(data) {
  i <- data$decoration == "dust::type"
  if (!any(i)) {
    return(NULL)
  }
  if (sum(i) > 1) {
    stop("More than one dust::type decoration found")
  }
  value <- data$params[[which(i)]]
  if (length(value) != 1L) {
    stop("Expected dust::type to have one argument")
  }
  if (nzchar(names(value))) {
    stop("Invalid format for [[dust::type()]] attribute")
  }
  as.character(value[[1]])
}


parse_metadata_param <- function(data) {
  i <- data$decoration == "dust::param"
  if (!any(i)) {
    return(NULL)
  }
  value <- lapply(data$params[i], parse_metadata_param1)

  nms <- vcapply(value, "[[", "name")
  if (any(duplicated(nms))) {
    stop("Duplicated [[dust::param()]] attributes: ",
         paste(squote(unique(nms[duplicated(nms)])), collapse = ", "))
  }

  set_names(lapply(value, "[[", "data"), nms)
}


parse_metadata_param1 <- function(x) {
  if (length(x) == 0) {
    stop("At least one argument required to [[dust::param]]")
  }
  if (nzchar(names(x)[[1]])) {
    stop("First argument of [[dust::param]] must be unnamed")
  }
  if (any(!nzchar(names(x)[-1]))) {
    stop("Arguments 2 and following of of [[dust::param]] must be named")
  }

  ## I think that we should allow only a restricted set here perhaps?
  ## Or a general set special case some like required/default
  list(name = as.character(x[[1]]),
       data = lapply(x[-1], function(el)
         if (is.symbol(el)) as.character(el) else el))
}


parse_metadata_error <- function(message, filename, line) {
  stop(sprintf("%s (%s:%s)", message, filename, line), call. = FALSE)
}


parse_metadata_guess_type <- function(txt) {
  re <- "^\\s*class\\s+([^{ ]+)\\s*(\\{.*|$)"
  i <- grep(re, txt)
  if (length(i) != 1L) {
    stop("Could not automatically detect class name; add [[dust::type]]?")
  }
  sub(re, "\\1", txt[[i]])
}
