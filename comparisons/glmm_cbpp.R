suppressPackageStartupMessages(library(lme4))

# 1. Load the dataset
data <- read.csv("tests/data/cbpp_binary.csv")
data$herd <- as.factor(data$herd)

# 2. Fit the model (Laplace / nAGQ = 1, lme4 default)
cat("\nFitting model: y ~ period2 + period3 + period4 + (1 | herd)\n")
# Using the pre-dummied period columns to match what we'll do in Rust/Python/Julia
fit <- glmer(y ~ period2 + period3 + period4 + (1 | herd),
             data = data,
             family = binomial(link = "logit"))

# Adaptive Gauss–Hermite (nAGQ = 7) — compare with Rust `glmer(..., 7)` and Python `n_agq=7`.
# Note: lme-rs estimates θ with Laplace then evaluates AGQ in the final PIRLS step; lme4 uses AGQ in the θ objective when nAGQ > 1.
fit_agq <- glmer(y ~ period2 + period3 + period4 + (1 | herd),
                 data = data,
                 family = binomial(link = "logit"),
                 nAGQ = 7)

# 3. Print fitted coefficients
cat("\n=== Fixed Effects (Laplace / nAGQ = 1) ===\n")
print(fixef(fit))

cat("\n=== Fixed Effects (AGQ, nAGQ = 7) ===\n")
print(fixef(fit_agq))

# 4. Generate Predictions (Response scale / Probabilities)
cat("\n=== Predictions (Probabilities) ===\n")
cat("Generating predictions for herd 1 across periods...\n")

newdata <- data.frame(
  herd = factor(c("1", "1", "1", "1"), levels = levels(data$herd)),
  period2 = c(0, 1, 0, 0),
  period3 = c(0, 0, 1, 0),
  period4 = c(0, 0, 0, 1)
)

# Predict population-level probabilities (re.form = NA, type = "response")
preds <- predict(fit_agq, newdata, re.form = NA, type = "response")
cat("Predictions:\n")
print(preds)
