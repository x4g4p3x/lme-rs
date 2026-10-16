# Matched random-slope references and independent t/F distribution checks.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: lmm_random_slope_v2.R RUN WORKER_INDEX WORKERS")
out <- args[[1]]
index <- as.integer(args[[2]])
workers <- as.integer(args[[3]])
if (index < 0 || index >= workers || workers < 1 || workers > 4) stop("invalid workers")
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
manifest <- jsonlite::read_json(file.path(out, "manifest.json"))
if (manifest$protocol != "lmm-random-slope-pilot-v2") stop("wrong protocol")
designs <- jsonlite::read_json(file.path(out, "designs.json"))

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

reference <- function(design, y) {
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
          REML = TRUE,
          control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
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
          beta = beta, sigma2 = sigma(fit)^2, covariance = covariance,
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

for (design in designs[seq(index + 1L, length(designs), by = workers)]) {
  folder <- file.path(out, "cases", design$id)
  target <- file.path(folder, "paired.jsonl")
  environment_path <- file.path(folder, "reference-environment.json")
  if (file.exists(target) || file.exists(environment_path)) stop("reference output exists")
  inputs <- gzfile(file.path(folder, "inputs.jsonl.gz"), "rt")
  candidates <- gzfile(file.path(folder, "candidate.jsonl.gz"), "rt")
  stream <- file(target, "wx")
  for (i in seq_len(manifest$replicates)) {
    input_line <- readLines(inputs, n = 1, warn = FALSE)
    candidate_line <- readLines(candidates, n = 1, warn = FALSE)
    if (length(input_line) != 1 || length(candidate_line) != 1) stop("missing input/candidate")
    row <- jsonlite::fromJSON(input_line, simplifyVector = FALSE)
    candidate <- jsonlite::fromJSON(candidate_line, simplifyVector = FALSE)
    if (row$case != design$id || candidate$case != design$id ||
      row$replicate != i - 1L || candidate$replicate != i - 1L) {
      stop("misaligned records")
    }
    ref <- reference(design, row$y)
    candidate$candidate$methods <- lapply(candidate$candidate$methods, enrich)
    value <- list(
      case = design$id, replicate = i - 1L,
      candidate_line_sha256 = digest::digest(candidate_line, algo = "sha256", serialize = FALSE),
      candidate = candidate$candidate, reference = ref
    )
    writeLines(jsonlite::toJSON(value, auto_unbox = TRUE, digits = 17, null = "null"), stream)
    flush(stream)
  }
  if (length(readLines(inputs, n = 1, warn = FALSE)) != 0 ||
    length(readLines(candidates, n = 1, warn = FALSE)) != 0) {
    stop("extra records")
  }
  close(inputs)
  close(candidates)
  close(stream)
  n <- manifest$replicates
  cp <- lapply(0:n, function(k) {
    c(
      if (k == 0) 0 else qbeta(0.025, k, n - k + 1),
      if (k == n) 1 else qbeta(0.975, k + 1, n - k)
    )
  })
  jsonlite::write_json(list(
    r_version = R.version.string,
    namespace_versions = setNames(
      lapply(sort(loadedNamespaces()), function(x) as.character(packageVersion(x))),
      sort(loadedNamespaces())
    ), cp95 = cp,
    gc_maximum_mib = sum(gc()[, 6])
  ), environment_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
  cat(sprintf("reference %d/%d: %s\n", index + 1L, workers, design$id))
}
