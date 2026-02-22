#!/usr/bin/env Rscript

# Create a local library directory to circumvent permission errors
dir.create("rlib", showWarnings = FALSE)
.libPaths(c("rlib", .libPaths()))

# Check and install lme4 and jsonlite if needed
if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4", repos = "https://cloud.r-project.org", lib = "rlib")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org", lib = "rlib")
}

library(lme4)
library(jsonlite)

# Simple intercept-only model: Reaction ~ 1 + (1 | Subject) using sleepstudy
data("sleepstudy", package = "lme4")
suppressWarnings(
  fm1 <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy, REML = TRUE)
)

# Extract components
X <- getME(fm1, "X")
Z <- getME(fm1, "Z")
Zt <- getME(fm1, "Zt")
y <- getME(fm1, "y")
theta <- getME(fm1, "theta")
beta <- fixef(fm1)
reml_crit <- lme4::REMLcrit(fm1)

out <- list(
  model = unbox("Reaction ~ 1 + (1 | Subject)"),
  inputs = list(
    X = as.matrix(X),
    Zt = as.matrix(Zt),
    y = as.numeric(y)
  ),
  outputs = list(
    theta = as.numeric(theta),
    beta = as.numeric(beta),
    reml_crit = unbox(as.numeric(reml_crit))
  )
)

# Save to JSON
dir.create("tests/data", recursive = TRUE, showWarnings = FALSE)
write_json(out, "tests/data/intercept_only.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write.csv(sleepstudy, "tests/data/sleepstudy.csv", row.names = FALSE)
cat("Successfully generated tests/data/intercept_only.json and sleepstudy.csv\n")
