# Selected post-study comparator diagnostics. Every fixed variant is retained.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: lmm_random_slope_reference_controls.R DIRECTORY")
out <- args[[1]]
if (file.exists(file.path(out, "attempts.jsonl"))) stop("output exists")
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(pbkrtest))
lock <- jsonlite::read_json("scientific/renv.lock")
if (as.character(getRversion()) != lock$R$Version) stop("R version drift")
for (name in names(lock$Packages)) {
  if (packageDescription(name, fields = "Version") != lock$Packages[[name]]$Version) {
    stop(paste("reference package drift", name))
  }
}
for (name in names(lock$LoadedNamespaces)) {
  if (as.character(packageVersion(name)) != lock$LoadedNamespaces[[name]]) {
    stop(paste("reference namespace drift", name))
  }
}
enrich <- function(method) {
  if (!is.null(method$error)) {
    return(method)
  }
  tryCatch(
    {
      if (is.null(method$df)) {
        critical <- qnorm(0.975)
        p <- 2 * pnorm(-abs(method$beta / method$se))
      } else {
        if (!is.numeric(method$df) || !is.finite(method$df) || method$df <= 0) stop("invalid df")
        critical <- qt(0.975, method$df)
        p <- pf(method$f, 1, method$df, lower.tail = FALSE)
      }
      if (!is.numeric(method$p) || !is.finite(method$p) || abs(p - method$p) > 1e-8) {
        stop("distribution p mismatch")
      }
      method$critical <- critical
      method$distribution_p <- p
      method
    },
    error = function(e) list(error = conditionMessage(e), raw = method)
  )
}

reference <- function(design, y, variant) {
  data <- data.frame(
    y = unlist(y), x = unlist(design$x),
    group = factor(sprintf("g%03d", unlist(design$group)))
  )
  before <- proc.time()[["elapsed"]]
  warnings <- character()
  result <- tryCatch(
    withCallingHandlers(
      {
        fit <- lmerTest::lmer(y ~ x + (1 + x | group), data,
          REML = TRUE, start = unlist(variant$start),
          control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = variant$control)
        )
        messages <- fit@optinfo$conv$lme4$messages
        problems <- messages[!grepl("boundary.*singular", messages)]
        converged <- length(problems) == 0 && all(fit@optinfo$conv$opt == 0)
        random_covariance <- as.matrix(lme4::VarCorr(fit)[[1]])
        if (!identical(rownames(random_covariance), c("(Intercept)", "x"))) stop("random covariance order differs")
        covariance <- unname(random_covariance)
        beta <- unname(lme4::fixef(fit)[[2]])
        se <- unname(sqrt(diag(vcov(fit)))[[2]])
        record <- list(
          beta = beta, sigma2 = sigma(fit)^2, covariance = covariance, theta = unname(lme4::getME(fit, "theta")),
          criterion = -2 * as.numeric(logLik(fit)), converged = converged,
          package_singular = lme4::isSingular(fit, tol = 1e-4),
          optimizer_messages = as.list(messages), optimizer_code = fit@optinfo$conv$opt,
          methods = list()
        )
        if (converged) {
          record$methods$normal <- list(
            beta = beta, se = se, df = NULL,
            f = (beta / se)^2, p = 2 * pnorm(-abs(beta / se))
          )
          record$methods$satterthwaite <- tryCatch(
            {
              test <- lmerTest::contest1D(fit, c(0, 1), ddf = "Satterthwaite")
              list(
                beta = beta, se = test[["Std. Error"]], df = test$df,
                f = test[["t value"]]^2, p = test[["Pr(>|t|)"]]
              )
            },
            error = function(e) list(error = conditionMessage(e))
          )
          record$methods$kenward_roger <- tryCatch(
            {
              variance <- unname(pbkrtest::vcovAdj(fit)[2, 2])
              test <- pbkrtest::KRmodcomp(fit, matrix(c(0, 1), nrow = 1))$stats
              if (abs(test$F.scaling - 1) > 1e-8 || abs(test$Fstat - beta^2 / variance) >
                1e-8 * max(1, test$Fstat)) {
                stop("scalar KR inconsistency")
              }
              list(
                beta = beta, se = sqrt(variance), df = test$ddf,
                f = test$Fstat, p = test$p.value, scaling = test$F.scaling
              )
            },
            error = function(e) list(error = conditionMessage(e))
          )
        } else {
          record$fit_error <- "nonconvergence"
        }
        record
      },
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) invokeRestart("muffleMessage")
    ),
    error = function(e) list(methods = list(), converged = FALSE, fit_error = conditionMessage(e))
  )
  result$warnings <- as.list(warnings)
  result$seconds <- proc.time()[["elapsed"]] - before
  result$methods <- lapply(result$methods, enrich)
  result
}


inputs <- jsonlite::read_json(file.path(out, "inputs.json"))
variants <- jsonlite::read_json(file.path(out, "variants.json"))
stream <- file(file.path(out, "attempts.jsonl"), "wx")
for (row in inputs) {
  for (variant in variants) {
    result <- reference(row$case, row$y, variant)
    writeLines(jsonlite::toJSON(list(
      case = row$case$id, replicate = row$replicate,
      variant = variant$id, fit = result
    ), auto_unbox = TRUE, digits = 17, null = "null"), stream)
    flush(stream)
  }
}
close(stream)
jsonlite::write_json(
  list(
    r_version = R.version.string,
    namespaces = setNames(
      lapply(sort(loadedNamespaces()), function(x) as.character(packageVersion(x))),
      sort(loadedNamespaces())
    )
  ), file.path(out, "reference-environment.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 17
)
