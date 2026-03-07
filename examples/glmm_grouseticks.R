library(lme4)

# 1. Load the dataset
data <- read.csv("tests/data/grouseticks.csv")

# 2. Fit the model
cat("\nFitting Poisson GLMM: TICKS ~ YEAR + HEIGHT + (1 | BROOD)\n")
fit <- glmer(TICKS ~ YEAR + HEIGHT + (1 | BROOD), data = data, family = poisson)

# 3. Print the summary
cat("\n=== Model Summary ===\n")
print(summary(fit))

# 4. Generate Predictions
cat("\n=== Predictions (Response Scale) ===\n")
cat("Generating expected tick counts for 3 new broods...\n")

newdata <- data.frame(
  YEAR = c(96, 96, 97),
  HEIGHT = c(400, 500, 450),
  BROOD = c("new1", "new2", "new3")
)

# Predict population-level (re.form = NA) and on response scale (type="response")
preds <- predict(fit, newdata, re.form = NA, type = "response")
cat("Expected Tick Counts:\n")
print(preds)
