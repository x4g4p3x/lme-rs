# Independent reference execution and distribution quantiles for frozen inputs.
# Candidate fits are never substituted with reference results.
args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(1L, 2L, 4L))) {
  stop("usage: lmm_calibration.R RUN_DIRECTORY [--worker IDX WORKERS | --finalize]")
}
out <- normalizePath(args[[1L]], mustWork = TRUE)
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
if (identical(manifest$namespace, "confirmatory")) {
  lock <- jsonlite::read_json("scientific/renv.lock", simplifyVector = TRUE)
  if (as.character(getRversion()) != lock$R$Version) stop("locked R version drift")
  for (pkg in names(lock$Packages)) {
    entry <- lock$Packages[[pkg]]
    raw <- suppressWarnings(packageDescription(pkg, fields = "Version"))
    if (is.na(raw) || !identical(as.character(raw), as.character(entry$Version))) {
      stop(paste("locked package raw version drift:", pkg))
    }
    if (!identical(as.character(packageVersion(pkg)), as.character(entry$NormalizedVersion))) {
      stop(paste("locked package version drift:", pkg))
    }
  }
  for (ns in names(lock$LoadedNamespaces)) {
    if (!identical(as.character(packageVersion(ns)), as.character(lock$LoadedNamespaces[[ns]]))) {
      stop(paste("locked namespace version drift:", ns))
    }
  }
}

designs <- read_json("designs.json")
designs <- setNames(designs, vapply(designs, function(x) x$id, character(1)))
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

fit_reference <- function(design, y) {
  data <- data.frame(
    y = unlist(y), x = unlist(design$x),
    group = factor(sprintf("g%03d", unlist(design$group)))
  )
  warnings <- character()
  before <- proc.time()[["elapsed"]]
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
  ref$methods <- lapply(ref$methods, enrich)
  ref
}

cp_matrix <- function(n, alpha) {
  if (n == 0L) {
    return(matrix(c(0, 1), nrow = 1L))
  }
  idx <- seq_len(n)
  rev_idx <- seq(n, 1L)
  cbind(
    c(0, qbeta(alpha / 2, idx, rev_idx)),
    c(qbeta(1 - alpha / 2, idx, rev_idx), 1)
  )
}

run_pilot <- function() {
  input <- file(file.path(out, "inputs.jsonl"), "r")
  candidate <- file(file.path(out, "candidate.jsonl"), "r")
  paired <- file(file.path(out, "paired.jsonl"), "wx")
  emit <- function(value) {
    writeLines(jsonlite::toJSON(value, auto_unbox = TRUE, digits = 17, null = "null"), paired)
    flush(paired)
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
    ref <- fit_reference(design, row$y)
    cand$methods <- lapply(cand$methods, enrich)
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
    namespace_versions = setNames(
      lapply(sort(loadedNamespaces()), function(x) as.character(packageVersion(x))),
      sort(loadedNamespaces())
    ),
    seconds = proc.time()[["elapsed"]] - started
  ), file.path(out, "reference-environment.json"), auto_unbox = TRUE, digits = 17, pretty = TRUE)
  writeLines(capture.output(sessionInfo()), file.path(out, "R-session.txt"))
}

run_confirmatory_worker <- function(worker_index, worker_count) {
  n_cand <- as.integer(manifest$replicates)
  n_ref <- as.integer(manifest$reference_replicates)
  if (worker_index < 1L || worker_index > worker_count) stop("invalid worker index")
  assigned <- designs[seq(worker_index, length(designs), by = worker_count)]
  for (design in assigned) {
    case_dir <- file.path(out, "cases", design$id)
    dist_path <- file.path(case_dir, "candidate-distributions.json")
    ref_path <- file.path(case_dir, "reference.jsonl")
    if (file.exists(dist_path) || file.exists(ref_path)) {
      stop(paste("existing reference output in", case_dir))
    }
    cand_con <- gzfile(file.path(case_dir, "candidate.jsonl.gz"), "rt")
    cand_lines <- readLines(cand_con, warn = FALSE)
    close(cand_con)
    if (length(cand_lines) != n_cand) stop(paste("wrong candidate count for", design$id))
    cands <- jsonlite::fromJSON(
      paste0("[", paste(cand_lines, collapse = ","), "]"),
      simplifyVector = FALSE
    )
    for (idx in seq_len(n_cand)) {
      if (!identical(cands[[idx]]$case, design$id) ||
        as.integer(cands[[idx]]$replicate) != (idx - 1L)) {
        stop(paste("candidate replicate order mismatch in", design$id))
      }
    }
    dist_out <- list()
    for (mname in c("normal", "satterthwaite", "kenward_roger")) {
      em <- lapply(cands, function(cand) {
        m <- cand$methods[[mname]]
        if (is.null(m)) list(error = "missing method") else enrich(m)
      })
      dist_out[[mname]] <- list(
        critical = vapply(
          em,
          function(x) if (is.null(x$critical)) NA_real_ else as.numeric(x$critical),
          numeric(1)
        ),
        distribution_p = vapply(
          em,
          function(x) if (is.null(x$distribution_p)) NA_real_ else as.numeric(x$distribution_p),
          numeric(1)
        ),
        error = vapply(
          em,
          function(x) if (is.null(x$error)) NA_character_ else as.character(x$error),
          character(1)
        )
      )
    }
    writeLines(
      jsonlite::toJSON(dist_out, auto_unbox = FALSE, digits = 17, na = "null"),
      dist_path,
      useBytes = TRUE
    )

    in_con <- gzfile(file.path(case_dir, "inputs.jsonl.gz"), "rt")
    in_lines <- readLines(in_con, n = n_ref, warn = FALSE)
    close(in_con)
    if (length(in_lines) != n_ref) stop(paste("missing reference inputs for", design$id))
    inputs <- jsonlite::fromJSON(
      paste0("[", paste(in_lines, collapse = ","), "]"),
      simplifyVector = FALSE
    )
    ref_con <- file(ref_path, "wx")
    for (idx in seq_len(n_ref)) {
      row <- inputs[[idx]]
      if (!identical(row$case, design$id) || as.integer(row$replicate) != (idx - 1L)) {
        close(ref_con)
        stop(paste("input replicate mismatch in", design$id))
      }
      ref <- fit_reference(design, row$y)
      record <- list(case = design$id, replicate = row$replicate, reference = ref)
      writeLines(
        jsonlite::toJSON(record, auto_unbox = TRUE, digits = 17, null = "null"),
        ref_con
      )
    }
    close(ref_con)
    cat(sprintf(
      "reference worker %d/%d completed %s (%d dist, %d fits)\n",
      worker_index, worker_count, design$id, n_cand, n_ref
    ))
  }
}

finalize_confirmatory <- function(started) {
  n_cand <- as.integer(manifest$replicates)
  n_ref <- as.integer(manifest$reference_replicates)
  for (design in designs) {
    case_dir <- file.path(out, "cases", design$id)
    if (!file.exists(file.path(case_dir, "candidate-distributions.json")) ||
      !file.exists(file.path(case_dir, "reference.jsonl"))) {
      stop(paste("missing reference artifacts for", design$id))
    }
  }
  bounds <- list(
    pointwise = cp_matrix(n_cand, 0.05),
    simultaneous = cp_matrix(n_cand, 0.05 / manifest$primary_family),
    reference_pointwise = cp_matrix(n_ref, 0.05),
    reference_simultaneous = cp_matrix(n_ref, 0.05 / manifest$primary_family)
  )
  writeLines(
    jsonlite::toJSON(bounds, auto_unbox = TRUE, digits = 17),
    file.path(out, "binomial-bounds.json"),
    useBytes = TRUE
  )
  jsonlite::write_json(list(
    r_version = R.version.string,
    namespace_versions = setNames(
      lapply(sort(loadedNamespaces()), function(x) as.character(packageVersion(x))),
      sort(loadedNamespaces())
    ),
    seconds = proc.time()[["elapsed"]] - started
  ), file.path(out, "reference-environment.json"), auto_unbox = TRUE, digits = 17, pretty = TRUE)
  writeLines(capture.output(sessionInfo()), file.path(out, "R-session.txt"))
}

if (identical(manifest$namespace, "pilot")) {
  if (length(args) != 1L) stop("pilot mode takes only RUN_DIRECTORY")
  run_pilot()
} else if (identical(manifest$namespace, "confirmatory")) {
  started <- proc.time()[["elapsed"]]
  if (length(args) == 1L) {
    run_confirmatory_worker(1L, 1L)
    finalize_confirmatory(started)
  } else if (length(args) == 4L && identical(args[[2L]], "--worker")) {
    run_confirmatory_worker(as.integer(args[[3L]]), as.integer(args[[4L]]))
  } else if (length(args) == 2L && identical(args[[2L]], "--finalize")) {
    finalize_confirmatory(started)
  } else {
    stop("invalid confirmatory arguments")
  }
} else {
  stop("unsupported manifest namespace")
}
