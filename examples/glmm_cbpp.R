library(lme4)

# 1. Load the dataset
data <- read.csv("tests/data/cbpp_binary.csv")
data$herd <- as.factor(data$herd)

# 2. Fit the model
cat("\nFitting model: y ~ period2 + period3 + period4 + (1 | herd)\n")
# Using the pre-dummied period columns to match what we'll do in Rust/Python/Julia
fit <- glmer(y ~ period2 + period3 + period4 + (1 | herd), 
             data = data, 
             family = binomial(link = "logit"))

# 3. Print the summary
cat("\n=== Model Summary ===\n")
print(summary(fit))

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
preds <- predict(fit, newdata, re.form = NA, type = "response")
cat("Predictions:\n")
print(preds)
