library(lme4)

# 1. Load the dataset
data <- read.csv("tests/data/dyestuff.csv")
data$Batch <- as.factor(data$Batch)

# 2. Fit the model
cat("\nFitting model: Yield ~ 1 + (1 | Batch)\n")
fit <- lmer(Yield ~ 1 + (1 | Batch), data = data, REML = TRUE)

# 3. Print the summary
cat("\n=== Model Summary ===\n")
print(summary(fit))
