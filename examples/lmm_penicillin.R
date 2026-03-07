library(lme4)

# 1. Load the dataset
data <- read.csv("tests/data/penicillin.csv")
data$plate <- as.factor(data$plate)
data$sample <- as.factor(data$sample)

# 2. Fit the model
cat("\nFitting model: diameter ~ 1 + (1 | plate) + (1 | sample)\n")
fit <- lmer(diameter ~ 1 + (1 | plate) + (1 | sample), data = data, REML = TRUE)

# 3. Print the summary
cat("\n=== Model Summary ===\n")
print(summary(fit))

# 4. Generate Predictions
cat("\n=== Predictions ===\n")
cat("Generating predictions for new plates and samples...\n")

newdata <- data.frame(
  plate = factor(c("a", "b", "c", "d"), levels = levels(data$plate)),
  sample = factor(c("A", "C", "E", "F"), levels = levels(data$sample))
)

# Predict population-level (re.form = NA)
preds <- predict(fit, newdata, re.form = NA)
cat("Predictions:\n")
print(preds)
