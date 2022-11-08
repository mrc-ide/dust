##' Prepare data for use with the `$set_data()` method. This is not
##' required for use but tries to simplify the most common use case
##' where you have a [data.frame] with some column indicating "dust
##' time step" (`name_time`), and other columns that might be use in
##' your `data_compare` function. Each row will be turned into a named
##' R list, which your `dust_data` function can then work with to get
##' this time-steps values. See Details for use with multi-pars
##' objects.
##'
##' Note that here "dust time step" (`name_time`) refers to the *dust*
##' time step (which will be a non-negative integer) and not the
##' rescaled value of time that you probably use within the model. See
##' [dust_generator] for more information.
##'
##' The data object as accepted by `data_set` must be a [list] and
##' each element must itself be a list with two elements; the dust
##' `time` at which the data applies and any R object that corresponds
##' to data at that point. We expect that most of the time this second
##' element will be a key-value list with scalar keys, but more
##' flexibility may be required.
##'
##' For multi-data objects, the final format is a bit more awkward;
##' each time step we have a list with elements `time`, `data_1`,
##' `data_2`, ..., `data_n` for `n` parameters. There are two ways of
##' creating this that might be useful: *sharing* the data across all
##' parameters and using some column as a grouping value.
##'
##' The behaviour here is driven by the `multi` argument;
##'
##' * `NULL`: (the default) do nothing; this creates an object that
##'   is suitable for use with a `pars_multi = FALSE` dust
##'   object.
##' * `<integer>` (e.g., multi = 3); share the data across 3 sets of
##'   parameters. This number must match the number of parameter sets
##'   that your dust object is created with
##' * `<column_name>` (e.g., multi = "country"); the name of a column
##'   within your data to split the data at. This column must be a
##'   factor, and that factor must have levels that map to integers 1,
##'   2, ..., n (e.g., `unique(as.integer(object[[multi]]))` returns
##'   the integers `1:n`).
##'
##' @title Process data for dust
##'
##' @param object An object, at this point must be a [data.frame]
##'
##' @param name_time The name of the data column within `object`; this
##'   column must be integer-like and every integer must be
##'   nonnegative and unique
##'
##' @param multi Control how to interpret data for multi-parameter
##'   dust object; see Details
##'
##' @return A list of dust time/data pairs that will be used for the
##'   compare function in a compiled model.  Each element is a list of
##'   length two or more where the first element is the time step and
##'   the subsequent elements are data for that time step.
##'
##' @export
##' @examples
##' d <- data.frame(time = seq(0, 50, by = 10), a = runif(6), b = runif(6))
##' dust::dust_data(d)
dust_data <- function(object, name_time = "time", multi = NULL) {
  assert_is(object, "data.frame")
  times <- object[[name_time]]
  if (is.null(times)) {
    stop(sprintf("'%s' is not a column in %s",
                 name_time, deparse(substitute(object))))
  }
  itimes <- as.integer(round(times))
  if (any(itimes < 0)) {
    stop(sprintf("All elements in column '%s' must be nonnegative", name_time))
  }
  if (any(abs(times - itimes) > sqrt(.Machine$double.eps))) {
    stop(sprintf("All elements in column '%s' must be integer-like", name_time))
  }

  rows <- lapply(seq_len(nrow(object)), function(i) as.list(object[i, ]))
  if (is.null(multi)) {
    ret <- Map(list, itimes, rows)
  } else if (is_integer_like(multi)) {
    ret <- Map(function(i, d) c(list(i), rep(list(d), multi)), itimes, rows)
  } else if (is.character(multi)) {
    group <- object[[multi]]
    if (is.null(group)) {
      stop(sprintf("'%s' is not a column in %s",
                   multi, deparse(substitute(object))))
    }
    if (!is.factor(group)) {
      stop(sprintf("Column '%s' must be a factor", multi))
    }
    itimes <- unname(split(itimes, group))
    if (length(unique(itimes)) != 1L) {
      stop("All groups must have the same time steps, in the same order")
    }
    itimes <- itimes[[1L]]
    rows_grouped <- unname(split(rows, group))
    ret <- lapply(seq_along(itimes), function(i) {
      c(list(itimes[[i]]), lapply(rows_grouped, "[[", i))
    })
  } else {
    stop("Invalid option for 'multi'; must be NULL, integer or character")
  }

  if (any(duplicated(itimes))) {
    stop(sprintf("All elements in column '%s' must be unique", name_time))
  }

  ret
}
