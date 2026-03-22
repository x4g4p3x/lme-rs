#!/usr/bin/env Rscript

library(lme4)

# Load data
data <- read.csv("tests/data/sleepstudy.csv")
cat("Loaded", nrow(data), "rows from tests/data/sleepstudy.csv\n\n")

# Fit ML model
cat("Fitting model: Reaction ~ Days + (Days | Subject)\n")
cat("Evaluating Maximum Likelihood (REML = FALSE)...\n")
fit <- lmer(Reaction ~ Days + (Days | Subject), data = data, REML = FALSE)

# Print summary
cat("\n=== Model Summary ===\n")
print(summary(fit))

# Predict
newdata <- data.frame(
  Days = c(0, 1, 5, 10),
  Subject = c(308, 308, 308, 308)
)
preds <- predict(fit, newdata, re.form = NA)
cat("\n=== Predictions ===\n")
print(preds)
