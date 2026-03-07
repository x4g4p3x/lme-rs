library(lme4)

# 1. Load the dataset
data <- read.csv("tests/data/pastes.csv")
data$batch <- as.factor(data$batch)
data$cask <- as.factor(data$cask)

# 2. Fit the model
cat("\nFitting model: strength ~ 1 + (1 | batch/cask)\n")
fit <- lmer(strength ~ 1 + (1 | batch/cask), data = data, REML = TRUE)

# 3. Print the summary
cat("\n=== Model Summary ===\n")
print(summary(fit))

# 4. Generate Predictions
cat("\n=== Predictions ===\n")
cat("Generating predictions for population-level (re.form = NA)...\n")

newdata <- data.frame(
  batch = factor(c("A", "B", "C"), levels = levels(data$batch)),
  cask = factor(c("a", "b", "c"), levels = levels(data$cask))
)

preds <- predict(fit, newdata, re.form = NA)
cat("Predictions:\n")
print(preds)
