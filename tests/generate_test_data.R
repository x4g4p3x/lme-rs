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
# Models
fm1 <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy, REML = TRUE)
fm2 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = TRUE)

get_model_data <- function(model_str, fit) {
  X <- getME(fit, "X")
  Z <- getME(fit, "Z")
  Zt <- getME(fit, "Zt")
  y <- getME(fit, "y")
  theta <- getME(fit, "theta")
  beta <- fixef(fit)
  reml_crit <- unbox(as.numeric(lme4::REMLcrit(fit)))
  
  list(
    model = unbox(model_str),
    inputs = list(
      X = as.matrix(X),
      Zt = as.matrix(Zt),
      y = as.numeric(y)
    ),
    outputs = list(
      theta = as.numeric(theta),
      beta = as.numeric(beta),
      reml_crit = reml_crit
    )
  )
}

out1 <- get_model_data("Reaction ~ 1 + (1 | Subject)", fm1)
out2 <- get_model_data("Reaction ~ Days + (Days | Subject)", fm2)

# Save to JSON
dir.create("tests/data", recursive = TRUE, showWarnings = FALSE)
write_json(out1, "tests/data/intercept_only.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out2, "tests/data/random_slopes.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write.csv(sleepstudy, "tests/data/sleepstudy.csv", row.names = FALSE)
cat("Successfully generated test data.\n")
