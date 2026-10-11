# Independent reference execution and distribution quantiles for frozen inputs.
# Candidate fits are never substituted with reference results.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: lmm_calibration.R RUN_DIRECTORY")
out <- normalizePath(args[[1]], mustWork = TRUE)
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(pbkrtest))
read_json <- function(name) jsonlite::read_json(file.path(out, name), simplifyVector = FALSE)
manifest <- read_json("manifest.json")
baseline <- jsonlite::read_json("scientific/cases.json")
for (name in names(baseline$reference_packages)) {
  if (as.character(packageVersion(name)) != baseline$reference_packages[[name]]) {
    stop(paste("reference package version drift:", name))
  }
}
if (getRversion() != "4.6.1") stop("R version drift")
designs <- read_json("designs.json")
designs <- setNames(designs, vapply(designs, function(x) x$id, character(1)))
input <- file(file.path(out, "inputs.jsonl"), "r")
candidate <- file(file.path(out, "candidate.jsonl"), "r")
paired <- file(file.path(out, "paired.jsonl"), "wx")
emit <- function(value) {
  writeLines(jsonlite::toJSON(value, auto_unbox = TRUE, digits = 17, null = "null"), paired)
  flush(paired)
}
normal_critical <- qnorm(0.975)
enrich <- function(method) {
  if (!is.null(method$error)) {
    return(method)
  }
  tryCatch(
    {
      if (is.null(method$df)) {
        method$critical <- normal_critical
        method$distribution_p <- 2 * pnorm(-abs(method$beta / method$se))
      } else {
        if (!is.numeric(method$df) || !is.finite(method$df) || method$df <= 0) stop("invalid df")
        method$critical <- qt(0.975, method$df)
        method$distribution_p <- pf(method$f, 1, method$df, lower.tail = FALSE)
      }
      if (!is.numeric(method$p) || !is.finite(method$p) ||
        abs(method$p - method$distribution_p) > 1e-8) {
        stop("distribution p inconsistency")
      }
      method
    },
    error = function(e) list(error = conditionMessage(e), raw = method)
  )
}
started <- proc.time()[["elapsed"]]
index <- 0L
repeat {
  line <- readLines(input, n = 1L, warn = FALSE)
  if (length(line) == 0L) break
  candidate_line <- readLines(candidate, n = 1L, warn = FALSE)
  if (length(candidate_line) != 1L) stop("missing candidate attempt")
  row <- jsonlite::fromJSON(line, simplifyVector = FALSE)
  cand <- jsonlite::fromJSON(candidate_line, simplifyVector = FALSE)
  if (row$case != cand$case || row$replicate != cand$replicate) stop("candidate input mismatch")
  design <- designs[[row$case]]
  data <- data.frame(
    y = unlist(row$y), x = unlist(design$x),
    group = factor(sprintf("g%03d", unlist(design$group)))
  )
  warnings <- character()
  before <- proc.time()[["elapsed"]]
  ref <- list(methods = list(), converged = FALSE)
  ref <- tryCatch(
    withCallingHandlers(
      {
        fit <- lmerTest::lmer(y ~ x + (1 | group),
          data = data, REML = TRUE,
          control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
        )
        messages <- fit@optinfo$conv$lme4$messages
        # Boundary/singular messages do not by themselves imply nonconvergence.
        problems <- messages[!grepl("boundary.*singular", messages)]
        converged <- length(problems) == 0L && all(fit@optinfo$conv$opt == 0)
        beta <- unname(lme4::fixef(fit)[[2]])
        se <- unname(sqrt(diag(vcov(fit)))[[2]])
        result <- list(
          methods = list(), converged = converged, singular = lme4::isSingular(fit, tol = 1e-4),
          beta = beta, sigma2 = sigma(fit)^2,
          tau2 = sigma(fit)^2 * unname(lme4::getME(fit, "theta")[[1]])^2,
          optimizer_messages = as.list(messages), optimizer_code = fit@optinfo$conv$opt
        )
        if (converged) {
          result$methods$normal <- list(
            beta = beta, se = se, df = NULL,
            f = (beta / se)^2, p = 2 * pnorm(-abs(beta / se))
          )
          result$methods$satterthwaite <- tryCatch(
            {
              test <- lmerTest::contest1D(fit, c(0, 1), ddf = "Satterthwaite")
              list(
                beta = beta, se = test[["Std. Error"]], df = test[["df"]],
                f = test[["t value"]]^2, p = test[["Pr(>|t|)"]]
              )
            },
            error = function(e) list(error = conditionMessage(e))
          )
          result$methods$kenward_roger <- tryCatch(
            {
              covariance <- pbkrtest::vcovAdj(fit)
              variance <- unname(covariance[2, 2])
              test <- pbkrtest::KRmodcomp(fit, matrix(c(0, 1), nrow = 1))$stats
              if (abs(test$F.scaling - 1) > 1e-8 || abs(test$Fstat - beta^2 / variance) >
                1e-8 * max(1, test$Fstat)) {
                stop("scalar KR covariance inconsistency")
              }
              list(
                beta = beta, se = sqrt(variance), df = test$ddf, f = test$Fstat,
                p = test$p.value, scaling = test$F.scaling
              )
            },
            error = function(e) list(error = conditionMessage(e))
          )
        } else {
          result$fit_error <- "nonconvergence"
        }
        result
      },
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) invokeRestart("muffleMessage")
    ),
    error = function(e) list(methods = list(), converged = FALSE, fit_error = conditionMessage(e))
  )
  ref$seconds <- proc.time()[["elapsed"]] - before
  ref$warnings <- as.list(warnings)
  cand$methods <- lapply(cand$methods, enrich)
  ref$methods <- lapply(ref$methods, enrich)
  emit(list(case = row$case, replicate = row$replicate, candidate = cand, reference = ref))
  index <- index + 1L
  if (index %% 100L == 0L) cat(sprintf("reference: %d/%d\n", index, manifest$attempted_datasets))
}
if (length(readLines(candidate, n = 1L, warn = FALSE)) != 0L) stop("extra candidate attempt")
close(input)
close(candidate)
close(paired)
if (index != manifest$attempted_datasets) stop("wrong attempted dataset count")
n <- manifest$replicates
cp <- function(k, alpha) {
  c(
    if (k == 0) 0 else qbeta(alpha / 2, k, n - k + 1),
    if (k == n) 1 else qbeta(1 - alpha / 2, k + 1, n - k)
  )
}
jsonlite::write_json(list(
  pointwise = lapply(0:n, cp, alpha = 0.05),
  simultaneous = lapply(0:n, cp, alpha = 0.05 / manifest$primary_family)
), file.path(out, "binomial-bounds.json"), auto_unbox = TRUE, digits = 17)
jsonlite::write_json(list(
  r_version = R.version.string,
  namespace_versions = setNames(lapply(sort(loadedNamespaces()), function(x) as.character(packageVersion(x))), sort(loadedNamespaces())),
  seconds = proc.time()[["elapsed"]] - started
), file.path(out, "reference-environment.json"), auto_unbox = TRUE, digits = 17, pretty = TRUE)
writeLines(capture.output(sessionInfo()), file.path(out, "R-session.txt"))
