# Independent raw-fit eligibility, inference/count, GLS and CP verification.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: lmm_random_slope_stream_check.R MANIFEST REPORT OUTPUT")
if (file.exists(args[[3]])) stop("output exists")
manifest <- jsonlite::read_json(args[[1]])
summary <- jsonlite::read_json(file.path(args[[2]], "summary.json"))
if (manifest$mode != "rehearsal" || summary$confirmation_claim) stop("not a rehearsal")
if (digest::digest(file = args[[1]], algo = "sha256") != summary$manifest_sha256 ||
  digest::digest(file = file.path(args[[2]], "audit.jsonl.gz"), algo = "sha256") != summary$audit_sha256) {
  stop("manifest/audit identity differs")
}
expected_cases <- vapply(manifest$cases, function(c) c$id, character(1))
if (anyDuplicated(expected_cases)) stop("duplicate design case")
count_keys <- vapply(summary$counts, function(r) paste(r$case, r$engine, r$method, sep = "|"), character(1))
expected_counts <- as.vector(outer(expected_cases, as.vector(outer(c("candidate", "reference"),
  c("normal", "satterthwaite", "kenward_roger"), paste,
  sep = "|"
)), paste, sep = "|"))
primary_keys <- vapply(summary$primary, function(r) paste(r$case, r$method, sep = "|"), character(1))
primary_cases <- vapply(Filter(function(c) isTRUE(c$primary_future), manifest$cases), function(c) c$id, character(1))
expected_primary <- as.vector(outer(primary_cases, c("satterthwaite", "kenward_roger"), paste, sep = "|"))
if (anyDuplicated(count_keys) || !setequal(count_keys, expected_counts) ||
  anyDuplicated(primary_keys) || !setequal(primary_keys, expected_primary) || length(expected_primary) != 72) {
  stop("claim/count inventory differs")
}
scalar <- function(x) is.numeric(x) && length(x) == 1 && is.finite(x)
covariance <- function(fit) matrix(unlist(fit$covariance), 2, byrow = TRUE)
rank_cov <- function(fit) {
  sigma <- covariance(fit)
  sum(eigen(sigma, symmetric = TRUE, only.values = TRUE)$values > 1e-8 * max(sum(diag(sigma)), fit$sigma2))
}
eligible <- function(fit) {
  if (!isTRUE(fit$converged) || !scalar(fit$beta) || !scalar(fit$criterion) ||
    !scalar(fit$sigma2) || fit$sigma2 <= 0 || length(unlist(fit$covariance)) != 4 ||
    any(!is.finite(unlist(fit$covariance)))) {
    return(FALSE)
  }
  sigma <- covariance(fit)
  scale <- max(sum(diag(sigma)), fit$sigma2)
  max(abs(sigma - t(sigma))) <= 1e-12 * scale &&
    min(eigen(sigma, symmetric = TRUE, only.values = TRUE)$values) >= -1e-12 * scale
}
numerical <- function(fit, check, balanced) {
  if (balanced && is.null(check)) stop("missing balanced oracle record")
  if (!balanced && !is.null(check)) stop("unexpected imbalanced oracle record")
  if (!eligible(fit)) {
    return(FALSE)
  }
  if (is.null(check)) {
    return(TRUE)
  } # no balanced global oracle for imbalanced designs
  oracle <- check$oracle
  scale <- max(1, fit$sigma2, oracle$sigma2, sum(diag(covariance(fit))), sum(diag(covariance(oracle))))
  abs(fit$criterion - oracle$criterion) <= 1e-6 + 1e-8 * abs(oracle$criterion) &&
    abs(fit$sigma2 - oracle$sigma2) <= 1e-6 * scale + 1e-3 * abs(oracle$sigma2) &&
    all(abs(covariance(fit) - covariance(oracle)) <= 1e-6 * scale + 1e-3 * abs(covariance(oracle))) &&
    rank_cov(fit) == rank_cov(oracle)
}
decision <- function(fit, method, truth, valid) {
  if (!valid) {
    return(NULL)
  }
  value <- fit$methods[[method]]
  if (is.null(value) || !is.null(value$error) ||
    !all(vapply(value[c("beta", "se", "f", "p")], scalar, logical(1))) ||
    value$se <= 0 || value$f < 0 || value$p < 0 || value$p > 1 ||
    abs(value$beta - fit$beta) > 1e-12 + 1e-12 * abs(fit$beta)) {
    return(NULL)
  }
  expected_f <- (value$beta / value$se)^2
  if (!is.finite(expected_f) || abs(value$f - expected_f) > max(1e-10, 1e-8 * abs(expected_f))) {
    return(NULL)
  }
  if (method == "normal") {
    critical <- qnorm(.975)
    p <- 2 * pnorm(-abs(value$beta / value$se))
  } else {
    if (!scalar(value$df) || value$df <= 0) {
      return(NULL)
    }
    critical <- qt(.975, value$df)
    p <- pf(value$f, 1, value$df, lower.tail = FALSE)
  }
  lower <- value$beta - critical * value$se
  upper <- value$beta + critical * value$se
  if (!scalar(critical) || critical <= 0 || !scalar(lower) || !scalar(upper) ||
    abs(value$p - p) > 1e-8) {
    return(NULL)
  }
  reject <- value$p < .05
  if (reject == (lower <= 0 && upper >= 0)) {
    return(NULL)
  }
  c(reject = reject, cover = lower <= truth && upper >= truth)
}
counts <- new.env(parent = emptyenv())
key <- function(case, engine, method) paste(case, engine, method, sep = "|")
for (row in summary$counts) assign(key(row$case, row$engine, row$method), c(attempted = 0, failed = 0, reject = 0, cover = 0), envir = counts)
designs <- setNames(manifest$cases, vapply(manifest$cases, function(c) c$id, character(1)))
variant_counts <- setNames(
  lapply(manifest$variants, function(v) {
    setNames(numeric(11), c(
      "attempted", "convergence_not_established", "optimizer_nonconvergence", "interrupted",
      "exception", "fit_error", "invalid_fit", "balanced_numerical_miss",
      "method_failed:normal", "method_failed:satterthwaite", "method_failed:kenward_roger"
    ))
  }),
  vapply(manifest$variants, function(v) v$id, character(1))
)
stream <- gzfile(file.path(args[[2]], "audit.jsonl.gz"), "rt")
seen <- setNames(integer(length(designs)), names(designs))
known <- c(attempted = 0, reject = 0, cover = 0)
max_gls_error <- 0
repeat {
  line <- readLines(stream, n = 1, warn = FALSE)
  if (length(line) == 0) break
  row <- jsonlite::fromJSON(line, simplifyVector = FALSE)
  case <- designs[[row$case]]
  if (is.null(case) || row$replicate != seen[[row$case]]) stop("raw case/key inventory differs")
  seen[[row$case]] <- seen[[row$case]] + 1L
  bank <- row$reference_bank
  balanced <- length(unique(unlist(case$sizes))) == 1
  if (row$replicate < manifest$reference_n) {
    if (length(bank) != 4) stop("reference bank incomplete")
    for (j in seq_along(bank)) {
      raw <- bank[[j]]
      valid <- numerical(raw, row$candidate_numerical, balanced)
      error <- raw$fit_error
      has_error <- !is.null(error) && nzchar(error)
      interrupted <- identical(error, "interrupted_after_durable_start")
      nonconvergence <- identical(error, "nonconvergence")
      variant_counts[[j]] <- variant_counts[[j]] + c(
        1, !isTRUE(raw$converged), nonconvergence,
        interrupted, has_error && !interrupted && !nonconvergence, has_error, !eligible(raw),
        balanced && eligible(raw) && !valid,
        vapply(c("normal", "satterthwaite", "kenward_roger"), function(m) is.null(decision(raw, m, case$beta, valid)), logical(1))
      )
    }
    choices <- which(vapply(bank, eligible, logical(1)))
    selected <- if (length(choices) == 0) NULL else choices[[which.min(vapply(bank[choices], function(f) f$criterion, numeric(1)))]] - 1
    if (!identical(selected, row$selected_reference) &&
      !(length(selected) == 1 && length(row$selected_reference) == 1 && selected == row$selected_reference)) {
      stop("reference selection differs")
    }
  } else if (length(bank) != 0) stop("unexpected reference subset")
  engines <- list(candidate = row$candidate)
  checks <- list(candidate = row$candidate_numerical)
  if (length(bank) != 0) {
    engines$reference <- if (is.null(row$selected_reference)) list() else bank[[row$selected_reference + 1]]
    checks$reference <- row$reference_numerical
  }
  for (engine in names(engines)) {
    fit <- engines[[engine]]
    valid <- numerical(fit, checks[[engine]], length(unique(unlist(case$sizes))) == 1)
    for (method in c("normal", "satterthwaite", "kenward_roger")) {
      name <- key(case$id, engine, method)
      count <- get(name, envir = counts)
      count[["attempted"]] <- count[["attempted"]] + 1
      d <- decision(fit, method, case$beta, valid)
      if (is.null(d)) count[["failed"]] <- count[["failed"]] + 1 else count[c("reject", "cover")] <- count[c("reject", "cover")] + d
      assign(name, count, envir = counts)
    }
  }
  # Independent known-variance groupwise GLS, never a fitted-variance estimator.
  information <- matrix(0, 2, 2)
  score <- c(0, 0)
  first <- 1L
  for (n in unlist(case$sizes)) {
    at <- seq.int(first, length.out = n)
    x <- unlist(case$x)[at]
    y <- unlist(row$y)[at]
    sxx <- sum(x^2)
    inv <- solve(matrix(unlist(case$sigma), 2, byrow = TRUE) + diag(c(1 / n, 1 / sxx)))
    information <- information + inv
    score <- score + inv %*% c(mean(y), sum(x * y) / sxx)
    first <- first + n
  }
  v <- solve(information)
  b <- as.numeric(v %*% score)[[2]]
  se <- sqrt(v[2, 2])
  max_gls_error <- max(max_gls_error, abs(b - row$known_beta), abs(se - row$known_se))
  if (max_gls_error > 1e-10) stop("known GLS mismatch")
  known <- known + c(1, 2 * pnorm(-abs(b / se)) < .05, abs(b - case$beta) <= qnorm(.975) * se)
}
close(stream)
for (case in manifest$cases) {
  n <- if (case$primary_future) manifest$primary_n else manifest$secondary_n
  if (seen[[case$id]] != n) stop("missing raw dataset")
}
for (row in summary$counts) {
  actual <- get(key(row$case, row$engine, row$method), envir = counts)
  if (any(actual != unlist(row[c("attempted", "failed", "reject", "cover")]))) stop("raw decision counts differ")
}
if (any(known != unlist(summary$known_variance_gls[c("attempted", "reject", "cover")]))) stop("known GLS counts differ")
if (!setequal(names(variant_counts), names(summary$reference_variant_attempts))) stop("variant count inventory differs")
for (name in names(variant_counts)) {
  expected <- variant_counts[[name]]
  actual <- unlist(summary$reference_variant_attempts[[name]])
  if (!setequal(names(expected), names(actual)) || any(actual[names(expected)] != expected)) stop("variant failure counts differ")
}
tail <- .05 / 144
errors <- numeric()
for (row in summary$primary) {
  if (length(unlist(row$bounds)) != 2 || !all(vapply(row$bounds, scalar, logical(1)))) stop("invalid primary bounds")
  count <- get(key(row$case, "candidate", row$method), envir = counts)
  n <- count[["attempted"]]
  k <- count[["reject"]]
  m <- count[["failed"]]
  bounds <- c(
    if (k == 0) 0 else qbeta(tail, k, n - k + 1),
    if (k + m == n) 1 else qbeta(tail, k + m + 1, n - k - m, lower.tail = FALSE)
  )
  if (n != row$n || k != row$k || m != row$m ||
    (bounds[[1]] >= .04 && bounds[[2]] <= .06) != row$rule_pass) {
    stop("primary decision differs")
  }
  errors <- c(errors, abs(bounds - unlist(row$bounds)))
}
if (length(summary$primary) != 72 || max(errors) > 1e-12) stop("primary family/bounds differ")
jsonlite::write_json(
  list(
    scope = "independent raw-fit count/distribution/selection/GLS/CP rehearsal check",
    confirmation_claim = FALSE, datasets = sum(seen), primary_claims_checked = 72,
    count_rows_checked = length(summary$counts), maximum_cp_error = max(errors),
    maximum_known_gls_error = max_gls_error, r_version = R.version.string,
    source_sha256 = digest::digest(file = "comparisons/scientific/lmm_random_slope_stream_check.R", algo = "sha256"),
    summary_sha256 = digest::digest(file = file.path(args[[2]], "summary.json"), algo = "sha256"),
    audit_sha256 = digest::digest(file = file.path(args[[2]], "audit.jsonl.gz"), algo = "sha256")
  ),
  args[[3]],
  auto_unbox = TRUE, pretty = TRUE, digits = 17
)
