#!/usr/bin/env Rscript
# Fit-only timings for nlmer / post-fit inference against lme4 (optional lmerTest).

args <- commandArgs(trailingOnly = TRUE)
case <- NA_character_
warmups <- 1L
repeats <- 5L
i <- 1L
while (i <= length(args)) {
  if (args[[i]] == "--case") {
    case <- args[[i + 1L]]
    i <- i + 2L
  } else if (args[[i]] == "--warmups") {
    warmups <- as.integer(args[[i + 1L]])
    i <- i + 2L
  } else if (args[[i]] == "--repeats") {
    repeats <- as.integer(args[[i + 1L]])
    i <- i + 2L
  } else {
    stop("unknown argument: ", args[[i]])
  }
}
if (is.na(case)) {
  stop("missing --case")
}

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
has_lmertest <- requireNamespace("lmerTest", quietly = TRUE)
if (has_lmertest) {
  suppressPackageStartupMessages(library(lmerTest))
}

summarize <- function(samples) {
  list(
    min_seconds = jsonlite::unbox(min(samples)),
    max_seconds = jsonlite::unbox(max(samples)),
    mean_seconds = jsonlite::unbox(mean(samples)),
    median_seconds = jsonlite::unbox(stats::median(samples))
  )
}

time_body <- function(warmups, repeats, body) {
  if (warmups > 0L) {
    for (k in seq_len(warmups)) {
      body()
    }
  }
  samples <- numeric(repeats)
  for (k in seq_len(repeats)) {
    samples[[k]] <- system.time(body())[["elapsed"]]
  }
  samples
}

emit <- function(implementation, family, formula, n_obs, samples) {
  payload <- list(
    implementation = jsonlite::unbox(implementation),
    case = jsonlite::unbox(case),
    family = jsonlite::unbox(family),
    formula = jsonlite::unbox(formula),
    n_obs = jsonlite::unbox(as.integer(n_obs)),
    warmups = jsonlite::unbox(as.integer(warmups)),
    repeats = jsonlite::unbox(as.integer(repeats)),
    samples_seconds = as.numeric(samples),
    summary = summarize(samples)
  )
  cat(jsonlite::toJSON(payload, pretty = TRUE, digits = NA), sep = "\n")
}

skip <- function(reason) {
  cat(jsonlite::toJSON(list(skipped = jsonlite::unbox(reason)), pretty = TRUE), sep = "\n")
}

if (case == "sleepstudy_lmer") {
  data <- read.csv("tests/data/sleepstudy.csv", stringsAsFactors = FALSE)
  data$Subject <- factor(data$Subject)
  formula <- "Reaction ~ Days + (Days | Subject)"
  samples <- time_body(warmups, repeats, function() {
    lme4::lmer(Reaction ~ Days + (Days | Subject), data, REML = TRUE)
  })
  emit("r_lme4", "lmm_fit", formula, nrow(data), samples)
} else if (case == "orange_nlmer") {
  data <- read.csv("tests/data/orange.csv", stringsAsFactors = FALSE)
  data$Tree <- factor(data$Tree)
  formula <- "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree"
  start <- c(Asym = 200, xmid = 725, scal = 350)
  samples <- time_body(warmups, repeats, function() {
    lme4::nlmer(
      circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym | Tree,
      data = data,
      start = start
    )
  })
  emit("r_lme4", "nlmm_fit", formula, nrow(data), samples)
} else if (case == "sleepstudy_satterthwaite") {
  if (!has_lmertest) {
    skip("R:sleepstudy_satterthwaite (lmerTest not installed)")
    quit(save = "no", status = 0)
  }
  data <- read.csv("tests/data/sleepstudy.csv", stringsAsFactors = FALSE)
  data$Subject <- factor(data$Subject)
  base <- lmerTest::lmer(Reaction ~ Days + (Days | Subject), data, REML = TRUE)
  formula <- "Reaction ~ Days + (Days | Subject)"
  samples <- time_body(warmups, repeats, function() {
    stats::anova(base, ddf = "Satterthwaite", type = 3)
  })
  emit("r_lmerTest", "post_fit_inference", formula, nrow(data), samples)
} else if (case == "sleepstudy_kenward_roger") {
  if (!has_lmertest) {
    skip("R:sleepstudy_kenward_roger (lmerTest not installed)")
    quit(save = "no", status = 0)
  }
  data <- read.csv("tests/data/sleepstudy.csv", stringsAsFactors = FALSE)
  data$Subject <- factor(data$Subject)
  base <- lmerTest::lmer(Reaction ~ Days + (Days | Subject), data, REML = TRUE)
  formula <- "Reaction ~ Days + (Days | Subject)"
  samples <- time_body(warmups, repeats, function() {
    stats::anova(base, ddf = "Kenward-Roger", type = 3)
  })
  emit("r_lmerTest", "post_fit_inference", formula, nrow(data), samples)
} else {
  stop("unknown case: ", case)
}
