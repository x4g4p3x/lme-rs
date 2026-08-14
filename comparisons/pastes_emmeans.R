#!/usr/bin/env Rscript
# Estimated marginal means for the pastes `cask` factor.
#
# Run from the repository root. If `emmeans` is installed in an isolated library,
# set LME_R_RLIB to that directory before invoking this script.

extra_lib <- Sys.getenv("LME_R_RLIB", unset = "")
if (nzchar(extra_lib)) {
  .libPaths(c(extra_lib, .libPaths()))
}

suppressPackageStartupMessages(library(emmeans))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lme4))

df <- read.csv("tests/data/pastes.csv")
df$cask <- factor(df$cask)
fit <- lmer(strength ~ cask + (1 | batch), df, REML = TRUE)

emm <- emmeans(fit, ~cask, lmer.df = "asymptotic")
means <- as.data.frame(summary(emm, infer = c(TRUE, FALSE), level = 0.95))
pairwise <- as.data.frame(
  summary(pairs(emm, adjust = "tukey", reverse = TRUE), infer = c(FALSE, TRUE))
)

out <- list(
  source = paste0(
    "R ", getRversion(), "; lme4 ", packageVersion("lme4"),
    "; emmeans ", packageVersion("emmeans"),
    "; asymptotic reference-grid inference"
  ),
  formula = "strength ~ cask + (1 | batch)",
  term = "cask",
  levels = as.character(means$cask),
  estimate = unname(means$emmean),
  std_error = unname(means$SE),
  df = unname(means$df),
  lower = unname(means$asymp.LCL),
  upper = unname(means$asymp.UCL),
  comparisons = as.character(pairwise$contrast),
  pair_estimate = unname(pairwise$estimate),
  pair_std_error = unname(pairwise$SE),
  pair_df = unname(pairwise$df),
  pair_z = unname(pairwise$z.ratio),
  pair_p_tukey = unname(pairwise$p.value)
)

cat(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE), "\n")
