##' Prepare data for use with the `$set_data()` method. This is not
##' required for use but tries to simplify the most common use case
##' where you have a [data.frame] with some column indicating "model
##' step" (`name_step`), and other columns that might be use in your
##' `data_compare` function. Each row will be turned into a named R
##' list, which your `dust_data` function can then work with to get
##' this time-steps values. See Details for use with multi-pars
##' objects.
##'
##' The data object as accepted by `data_set` must be a [list] and
##' each element must itself be a list with two elements; the model
##' `step` at which the data applies and any R object that corresponds
##' to data at that point. We expect that most of the time this second
##' element will be a key-value list with scalar keys, but more
##' flexibility may be required.
##'
##' For multi-data objects, the final format is a bit more awkward;
##' each time step we have a list with elements `step`, `data_1`,
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
##'   within your data to split the data at. This columm must be a
##'   factor, and that factor must have levels that map to integers 1,
##'   2, ..., n (e.g., `unique(as.integer(object[[multi]]))` returns
##'   the integers `1:n`).
##'
##' @title Process data for dust
##'
##' @param object An object, at this point must be a [data.frame]
##'
##' @param name_step The name of the data column within `object`; this
##'   column must be integer-like and every integer must be
##'   nonnegative and unique
##'
##' @param multi Control how to interpret data for multi-parameter
##'   dust object; see Details
##'
##' @export
##' @examples
##' d <- data.frame(step = seq(0, 50, by = 10), a = runif(6), b = runif(6))
##' dust::dust_data(d)
dust_data <- function(object, name_step = "step", multi = NULL) {
  assert_is(object, "data.frame")
  steps <- object[[name_step]]
  if (is.null(steps)) {
    stop(sprintf("'%s' is not a column in %s",
                 name_step, deparse(substitute(object))))
  }
  isteps <- as.integer(round(steps))
  if (any(isteps < 0)) {
    stop(sprintf("All elements in column '%s' must be nonnegative", name_step))
  }
  if (any(abs(steps - isteps) > sqrt(.Machine$double.eps))) {
    stop(sprintf("All elements in column '%s' must be integer-like", name_step))
  }

  rows <- lapply(seq_len(nrow(object)), function(i) as.list(object[i, ]))
  if (is.null(multi)) {
    ret <- Map(list, isteps, rows)
  } else if (is_integer_like(multi)) {
    ret <- Map(function(i, d) c(list(i), rep(list(d), multi)), isteps, rows)
  } else if (is.character(multi)) {
    group <- object[[multi]]
    if (is.null(group)) {
      stop(sprintf("'%s' is not a column in %s",
                   multi, deparse(substitute(object))))
    }
    if (!is.factor(group)) {
      stop(sprintf("Column '%s' must be a factor", multi))
    }
    isteps <- unname(split(isteps, group))
    if (length(unique(isteps)) != 1L) {
      stop("All groups must have the same time steps, in the same order")
    }
    isteps <- isteps[[1L]]
    rows_grouped <- unname(split(rows, group))
    ret <- lapply(seq_along(isteps), function(i)
      c(list(isteps[[i]]), lapply(rows_grouped, "[[", i)))
  } else {
    stop("Invalid option for 'multi'; must be NULL, integer or character")
  }

  if (any(duplicated(isteps))) {
    stop(sprintf("All elements in column '%s' must be unique", name_step))
  }

  ret
}
