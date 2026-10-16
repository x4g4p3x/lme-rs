# Independent beta-quantile/count and binomial-tail check of Python planning.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: lmm_random_slope_planning.R INPUT OUTPUT")
if (file.exists(args[[2]])) stop("output exists")
input <- jsonlite::read_json(args[[1]])
cfg <- input$inputs
tail <- cfg$familywise_alpha / (2 * cfg$family)
band <- unlist(cfg$rejection_band)
counts <- function(n) {
  # Search beta quantiles, independently of the planner's binomial-tail search.
  lower <- function(k) if (k == 0) 0 else qbeta(tail, k, n - k + 1)
  upper <- function(k) if (k == n) 1 else qbeta(tail, k + 1, n - k, lower.tail = FALSE)
  lo <- 0L
  hi <- n + 1L
  while (lo < hi) {
    mid <- floor((lo + hi) / 2)
    if (mid == n + 1 || lower(mid) >= band[[1]]) hi <- mid else lo <- mid + 1
  }
  a <- lo
  lo <- -1L
  hi <- n
  while (hi - lo > 1) {
    mid <- floor((lo + hi) / 2)
    if (upper(mid) <= band[[2]]) lo <- mid else hi <- mid
  }
  c(a, if (band[[2]] == 1) n else lo)
}
errors <- numeric()
for (row in c(input$ideal, input$sensitivity)) {
  complete <- counts(row$n)
  observed <- c(max(0, complete[[1]]), min(row$valid, complete[[2]] - row$failed))
  if (!identical(as.numeric(complete), as.numeric(unlist(row$acceptance_complete_counts))) ||
    !identical(as.numeric(observed), as.numeric(unlist(row$acceptance_observed_counts)))) {
    stop("count mismatch")
  }
  q <- if (observed[[1]] > observed[[2]]) {
    1
  } else {
    pbinom(observed[[1]] - 1, row$valid, row$p_valid) +
      pbinom(observed[[2]], row$valid, row$p_valid, lower.tail = FALSE)
  }
  errors <- c(errors, abs(c(q, 1 - q, max(0, 1 - cfg$family * q)) - c(
    row$single_claim_failure_probability,
    row$single_claim_decision_power, row$joint_decision_power_lower_bound
  )))
}
if (max(errors) > 1e-10) stop("probability mismatch")
jsonlite::write_json(list(
  scope = "independent planning calculation; no model fits or calibration claim",
  input_sha256 = digest::digest(file = args[[1]], algo = "sha256"),
  source_sha256 = digest::digest(file = "comparisons/scientific/lmm_random_slope_planning.R", algo = "sha256"),
  r_version = R.version.string,
  jsonlite_version = as.character(packageVersion("jsonlite")),
  rows_checked = length(input$ideal) + length(input$sensitivity),
  maximum_absolute_probability_error = max(errors), all_count_intervals_match = TRUE
), args[[2]], auto_unbox = TRUE, pretty = TRUE, digits = 17)
