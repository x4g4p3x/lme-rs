#!/usr/bin/env Rscript
# Golden LMM fit matrix for Wilkinson / RE edge cases vs lme4.
#
# Writes tests/data/r_edge_case_matrix.json. Run from the repository root:
#   Rscript tests/generate_r_edge_case_matrix.R

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
  library(splines)
})

scalar_check <- function(name, value, tolerance) {
  list(
    name = jsonlite::unbox(name),
    value = jsonlite::unbox(as.numeric(value)),
    tolerance = jsonlite::unbox(as.numeric(tolerance))
  )
}

named_checks <- function(values, tolerance, prefix = "") {
  nms <- names(values)
  if (is.null(nms)) {
    nms <- paste0(prefix, seq_along(values))
  }
  lapply(seq_along(values), function(i) {
    scalar_check(nms[[i]], values[[i]], tolerance)
  })
}

theta_named_checks <- function(fit, tolerance) {
  theta <- as.numeric(getME(fit, "theta"))
  cnms <- getME(fit, "cnms")
  labels <- character(0)
  # Iterate by index: || produces two cnms entries that share the grouping name.
  for (block in seq_along(cnms)) {
    grp <- names(cnms)[[block]]
    effects <- cnms[[block]]
    k <- length(effects)
    for (i in seq_len(k)) {
      for (j in seq_len(i)) {
        if (i == j) {
          labels <- c(labels, paste0(grp, ".", effects[[i]]))
        } else {
          labels <- c(labels, paste0(grp, ".", effects[[i]], ".", effects[[j]]))
        }
      }
    }
  }
  if (length(labels) != length(theta)) {
    labels <- paste0("theta_", seq_along(theta))
  }
  lapply(seq_along(theta), function(i) {
    scalar_check(labels[[i]], theta[[i]], tolerance)
  })
}

dump_lmm <- function(id, description, formula, data_path, data, reml,
                     coef_tol, theta_tol, sigma_tol, dev_tol, call_str) {
  fit <- lmer(formula, data = data, REML = reml)
  beta <- fixef(fit)
  objective <- as.numeric(if (isREML(fit)) lme4::REMLcrit(fit) else deviance(fit))
  list(
    id = jsonlite::unbox(id),
    description = jsonlite::unbox(description),
    kind = jsonlite::unbox("lmm"),
    data_path = jsonlite::unbox(data_path),
    formula = jsonlite::unbox(deparse1(formula)),
    reml = jsonlite::unbox(reml),
    reference = list(
      engine = jsonlite::unbox("lme4::lmer"),
      call = jsonlite::unbox(call_str),
      r_version = jsonlite::unbox(paste(R.version$version.string)),
      lme4_version = jsonlite::unbox(as.character(utils::packageVersion("lme4")))
    ),
    expected = list(
      coefficients = named_checks(beta, coef_tol),
      theta = theta_named_checks(fit, theta_tol),
      sigma2 = scalar_check("sigma2", sigma(fit)^2, sigma_tol),
      deviance = scalar_check(
        if (reml) "REML criterion" else "deviance",
        objective,
        dev_tol
      ),
      fitted = as.numeric(fitted(fit)),
      fitted_tolerance = jsonlite::unbox(0.05)
    )
  )
}

sleepstudy <- read.csv("tests/data/sleepstudy.csv", stringsAsFactors = FALSE)
sleepstudy$Subject <- factor(sleepstudy$Subject)

pastes <- read.csv("tests/data/pastes.csv", stringsAsFactors = FALSE)
pastes$batch <- factor(pastes$batch)
pastes$cask <- factor(pastes$cask)

cases <- list(
  dump_lmm(
    "sleepstudy_independent_slopes_reml",
    "Independent random intercept and slope via Wilkinson || (lme4 expands to two RE blocks).",
    Reaction ~ Days + (Days || Subject),
    "tests/data/sleepstudy.csv",
    sleepstudy,
    TRUE,
    0.05, 0.02, 0.15, 0.15,
    "lmer(Reaction ~ Days + (Days || Subject), sleepstudy, REML = TRUE)"
  ),
  dump_lmm(
    "sleepstudy_quadratic_i_reml",
    "Identity-protected quadratic I(Days^2) with a random intercept.",
    Reaction ~ Days + I(Days^2) + (1 | Subject),
    "tests/data/sleepstudy.csv",
    sleepstudy,
    TRUE,
    0.05, 0.02, 0.15, 0.15,
    "lmer(Reaction ~ Days + I(Days^2) + (1 | Subject), sleepstudy, REML = TRUE)"
  ),
  dump_lmm(
    "sleepstudy_sqrt_days_reml",
    "Unary sqrt(Days) transform (Days includes 0, so log() is undefined).",
    Reaction ~ sqrt(Days) + (1 | Subject),
    "tests/data/sleepstudy.csv",
    sleepstudy,
    TRUE,
    0.05, 0.02, 0.15, 0.15,
    "lmer(Reaction ~ sqrt(Days) + (1 | Subject), sleepstudy, REML = TRUE)"
  ),
  dump_lmm(
    "sleepstudy_poly_raw_reml",
    "Raw polynomial poly(Days, 2, raw = TRUE) plus a random intercept.",
    Reaction ~ poly(Days, 2, raw = TRUE) + (1 | Subject),
    "tests/data/sleepstudy.csv",
    sleepstudy,
    TRUE,
    0.05, 0.02, 0.15, 0.15,
    "lmer(Reaction ~ poly(Days, 2, raw = TRUE) + (1 | Subject), sleepstudy, REML = TRUE)"
  ),
  dump_lmm(
    "sleepstudy_poly_orthogonal_reml",
    "Orthogonal poly(Days, 2) plus a random intercept (column space of stats::poly).",
    Reaction ~ poly(Days, 2) + (1 | Subject),
    "tests/data/sleepstudy.csv",
    sleepstudy,
    TRUE,
    0.08, 0.03, 0.2, 0.2,
    "lmer(Reaction ~ poly(Days, 2) + (1 | Subject), sleepstudy, REML = TRUE)"
  ),
  dump_lmm(
    "sleepstudy_ns_df3_reml",
    "Natural cubic spline ns(Days, 3) plus a random intercept (column space of splines::ns).",
    Reaction ~ ns(Days, 3) + (1 | Subject),
    "tests/data/sleepstudy.csv",
    sleepstudy,
    TRUE,
    0.15, 0.05, 0.25, 0.25,
    "lmer(Reaction ~ ns(Days, 3) + (1 | Subject), sleepstudy, REML = TRUE)"
  ),
  dump_lmm(
    "pastes_no_intercept_cask_reml",
    "Cell-means coding: strength ~ 0 + cask with a batch random intercept.",
    strength ~ 0 + cask + (1 | batch),
    "tests/data/pastes.csv",
    pastes,
    TRUE,
    0.05, 0.02, 0.15, 0.15,
    "lmer(strength ~ 0 + cask + (1 | batch), pastes, REML = TRUE)"
  ),
  dump_lmm(
    "pastes_nested_slash_reml",
    "Nested grouping via slash: strength ~ 1 + (1 | batch/cask).",
    strength ~ 1 + (1 | batch / cask),
    "tests/data/pastes.csv",
    pastes,
    TRUE,
    0.05, 0.02, 0.15, 0.15,
    "lmer(strength ~ 1 + (1 | batch/cask), pastes, REML = TRUE)"
  )
)

payload <- list(
  schema_version = jsonlite::unbox(1L),
  generator = jsonlite::unbox("tests/generate_r_edge_case_matrix.R"),
  assessed_on = jsonlite::unbox(as.character(Sys.Date())),
  reference_environment = list(
    engine = jsonlite::unbox("R"),
    r_version = jsonlite::unbox(paste(R.version$version.string)),
    platform = jsonlite::unbox(R.version$platform),
    packages = list(
      lme4 = jsonlite::unbox(as.character(utils::packageVersion("lme4"))),
      splines = jsonlite::unbox(as.character(utils::packageVersion("splines"))),
      jsonlite = jsonlite::unbox(as.character(utils::packageVersion("jsonlite")))
    ),
    notes = list(
      jsonlite::unbox("Fit-level parity for Wilkinson / RE edge cases not covered by the main golden manifest."),
      jsonlite::unbox("Orthogonal poly() / ns() compare coefficients in the R column space; sign conventions may require the Rust test to align columns.")
    )
  ),
  cases = cases
)

out_path <- "tests/data/r_edge_case_matrix.json"
write(jsonlite::toJSON(payload, pretty = TRUE, digits = NA, auto_unbox = FALSE), out_path)
cat("Wrote ", out_path, " (", length(cases), " cases)\n", sep = "")

for (case in cases) {
  cat("\n== ", case$id, " ==\n", sep = "")
  cat("formula: ", case$formula, "\n", sep = "")
  cat("coef names: ", paste(vapply(case$expected$coefficients, function(x) x$name, ""), collapse = ", "), "\n", sep = "")
  cat("theta names: ", paste(vapply(case$expected$theta, function(x) x$name, ""), collapse = ", "), "\n", sep = "")
}
