suppressPackageStartupMessages(library(lme4))
source("comparisons/r_setup.R")

file_path <- file.path("tests", "data", "sleepstudy.csv")
if (!file.exists(file_path)) {
  stop("Could not find the dataset at ", file_path, ". Run from the repository root.")
}

data <- read.csv(file_path)

cat("\nFitting Gaussian GLMM: Reaction ~ Days + (1 | Subject)\n")
cat("(lme4::lmer — equivalent to glmer(gaussian(identity)))\n")

fit <- lmer(Reaction ~ Days + (1 | Subject), data = data, REML = TRUE)

cat("\n=== Model Summary ===\n")
print(summary(fit))

cat("\n=== Predictions (population-level, response scale) ===\n")
newdata <- data.frame(
  Days = c(0, 1),
  Subject = factor(c("308", "308"), levels = levels(factor(data$Subject)))
)
preds <- predict(fit, newdata, re.form = NA, type = "response")
cat("Predictions:\n")
print(preds)
