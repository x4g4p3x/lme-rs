suppressPackageStartupMessages(library(lme4))

# 1. Load the dataset
data <- read.csv("tests/data/sleepstudy.csv")

# 2. Fit the model
cat("\nFitting model: Reaction ~ Days + (Days | Subject)\n")
fit <- lmer(Reaction ~ Days + (Days | Subject), data = data, REML = TRUE)

# 3. Print fitted coefficients
cat("\n=== Fixed Effects ===\n")
print(fixef(fit))

# 4. Generate Predictions
cat("\n=== Predictions ===\n")
cat("Generating predictions for Subject 308 at Days 0, 1, 5, and 10...\n")

newdata <- data.frame(
  Days = c(0, 1, 5, 10),
  Subject = c("308", "308", "308", "308")
)

# Predict population-level (re.form = NA)
preds <- predict(fit, newdata, re.form = NA)
cat("Predictions:\n")
print(preds)
