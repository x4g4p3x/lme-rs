#!/usr/bin/env Rscript
# Same three figures as plot_demo.py, using lme4::lmer / glmer.
# Pixel sizes match figure_specs.py (DPI = 150).
#
# Usage (from repository root):
#   Rscript python/examples/plotting_demo/plot_r.R
# Or pass the repo root explicitly:
#   Rscript python/examples/plotting_demo/plot_r.R "C:/path/to/lme-rs"

suppressPackageStartupMessages({
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1) {
  normalizePath(args[1])
} else if (nzchar(Sys.getenv("LME_RS_ROOT", ""))) {
  normalizePath(Sys.getenv("LME_RS_ROOT"))
} else {
  normalizePath(getwd())
}

data_dir <- file.path(repo_root, "tests", "data")
out_dir <- file.path(repo_root, "python", "examples", "plotting_demo", "figures_r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fig_data <- file.path(repo_root, "python", "examples", "plotting_demo", "figures_data")
dir.create(fig_data, recursive = TRUE, showWarnings = FALSE)

DPI <- 150L
# figure_specs.py: px = inches * 150
W1 <- 975L
H1 <- 750L
W2 <- 1125L
H2 <- 750L

# --- Sleepstudy LMM ----------------------------------------------------------
df <- read.csv(file.path(data_dir, "sleepstudy.csv"), stringsAsFactors = FALSE)
df$Subject <- factor(df$Subject, levels = unique(as.character(df$Subject)))

fit <- lmer(Reaction ~ Days + (Days | Subject), data = df, REML = TRUE)

fi <- fitted(fit)
re <- residuals(fit)

png(file.path(out_dir, "sleepstudy_residuals_vs_fitted.png"),
    width = W1, height = H1, units = "px", res = DPI)
# Margins aligned with matplotlib figure_specs.MPL_ADJUST_RESIDUAL (approx.)
par(mar = c(5.2, 4.5, 4.5, 2.2))
plot(
  fi, re,
  pch = 16,
  col = grDevices::adjustcolor("#1F77B4", alpha.f = 0.35),
  cex = 0.4,
  xlab = "Fitted (conditional)",
  ylab = "Residual (y \u2212 fitted)",
  main = "Sleepstudy LMM \u2014 residual vs fitted"
)
abline(h = 0, col = "gray50", lty = 2, lwd = 0.8)
dev.off()

write.csv(
  data.frame(fitted = fi, residual = re),
  file.path(fig_data, "sleepstudy_residuals_r.csv"),
  row.names = FALSE
)

days <- df$Days
reaction <- df$Reaction
subjects <- unique(df$Subject)
idx <- seq(1L, length(subjects), by = 3L)
subj <- as.character(subjects[1L])

grid_days <- seq(min(days), max(days), length.out = 40L)
newdata <- data.frame(
  Days = grid_days,
  Subject = factor(subj, levels = levels(df$Subject))
)
line_pop <- predict(fit, newdata, re.form = NA)
line_cond <- predict(fit, newdata, re.form = NULL)

png(file.path(out_dir, "sleepstudy_days_reaction_curves.png"),
    width = W2, height = H2, units = "px", res = DPI)
# Extra top margin for legend (cf. MPL_ADJUST_SPAGHETTI).
par(mar = c(5.2, 4.5, 6.5, 2.2))
plot(
  NA,
  xlim = range(days),
  ylim = range(reaction),
  xlab = "Days",
  ylab = "Reaction (ms)",
  main = "Sleepstudy \u2014 observed trajectories and prediction curves"
)
for (s in subjects[idx]) {
  m <- df$Subject == s
  lines(days[m], reaction[m], col = gray(0.75), lwd = 0.9)
}
points(
  days, reaction,
  pch = 16,
  col = grDevices::adjustcolor("black", alpha.f = 0.25),
  cex = 0.35
)
lines(grid_days, line_pop, col = "#1F77B4", lwd = 2.2)
lines(grid_days, line_cond, col = "#FF7F0E", lwd = 2.0, lty = 2)
legend(
  "topleft",
  legend = c(
    "Population (\u00d7\u03b2)",
    paste0("Conditional (\u00d7\u03b2+Zb), Subject ", subj)
  ),
  col = c("#1F77B4", "#FF7F0E"),
  lty = c(1, 2),
  lwd = c(2.2, 2.0),
  bty = "n"
)
dev.off()

write.csv(
  data.frame(Days = grid_days, line_pop = line_pop, line_cond = line_cond),
  file.path(fig_data, "sleepstudy_curves_r.csv"),
  row.names = FALSE
)

# --- Grouseticks Poisson GLMM ------------------------------------------------
gt <- read.csv(file.path(data_dir, "grouseticks.csv"), stringsAsFactors = FALSE)
gt$BROOD <- factor(gt$BROOD, levels = unique(as.character(gt$BROOD)))

gfit <- glmer(
  TICKS ~ YEAR + HEIGHT + (1 | BROOD),
  data = gt,
  family = poisson,
  nAGQ = 1L
)
mu <- predict(gfit, type = "response", re.form = NA)
y <- gt$TICKS

png(file.path(out_dir, "grouseticks_observed_vs_fitted.png"),
    width = W1, height = H1, units = "px", res = DPI)
par(mar = c(5.2, 4.5, 4.5, 2.2))
max_val <- max(y, max(mu) * 1.05) + 1
xr <- range(c(0, mu, max_val))
yr <- range(c(0, y, max_val))
plot(
  NA,
  xlim = xr,
  ylim = yr,
  xlab = "Fitted expected count (population, response scale)",
  ylab = "Observed TICKS",
  main = "Grouseticks Poisson GLMM \u2014 observed vs fitted"
)
lines(c(0, max_val), c(0, max_val), col = "gray50", lty = 2, lwd = 0.9)
points(
  mu, y,
  pch = 16,
  col = grDevices::adjustcolor("#2CA02C", alpha.f = 0.35),
  cex = 0.4
)
legend(
  "topleft",
  legend = expression(y == hat(mu)),
  col = "gray50",
  lty = 2,
  lwd = 0.9,
  bty = "n"
)
dev.off()

write.csv(
  data.frame(mu = mu, y = y),
  file.path(fig_data, "grouseticks_r.csv"),
  row.names = FALSE
)

cat("Wrote R figures to:\n")
cat(" ", file.path(out_dir, "sleepstudy_residuals_vs_fitted.png"), "\n", sep = "")
cat(" ", file.path(out_dir, "sleepstudy_days_reaction_curves.png"), "\n", sep = "")
cat(" ", file.path(out_dir, "grouseticks_observed_vs_fitted.png"), "\n", sep = "")
