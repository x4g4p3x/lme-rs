# Planning only: ideal true rejection probability 0.05, no failed fits.
# This does not estimate the candidate's operating characteristics.
family <- 48L
for (n in c(10000L, 20000L, 25000L)) {
  k <- seq(floor(0.04 * n), ceiling(0.06 * n))
  lower <- qbeta(0.05 / family / 2, k, n - k + 1)
  upper <- qbeta(1 - 0.05 / family / 2, k + 1, n - k)
  eligible <- k[lower >= 0.04 & upper <= 0.06]
  probability <- if (length(eligible)) {
    pbinom(max(eligible), n, 0.05) - pbinom(min(eligible) - 1, n, 0.05)
  } else {
    0
  }
  # Union bound does not assume independence between methods or cases.
  cat(sprintf(
    "n=%d eligible=%d..%d single_claim=%.9f joint48_lower_bound=%.9f\n",
    n, min(eligible), max(eligible), probability, max(0, 1 - family * (1 - probability))
  ))
}
