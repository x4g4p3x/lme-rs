suppressPackageStartupMessages(library(lme4))

file_path <- file.path("tests", "data", "sleepstudy.csv")
if (!file.exists(file_path)) {
  stop("Could not find the dataset at ", file_path, ". Run from the repository root.")
}

data <- read.csv(file_path)
n <- nrow(data)
# Same prior weights as Rust `lmer_weighted` example / `benches/bench_math`
data$w <- 0.5 + ((seq_len(n) - 1) %% 5) * 0.1

cat("\nFitting weighted model: Reaction ~ Days + (Days | Subject)\n")
cat("Prior weights w_i = 0.5 + (row_index mod 5) * 0.1\n")

fit <- lmer(Reaction ~ Days + (Days | Subject), data = data, weights = w, REML = TRUE)

cat("\n=== Model Summary ===\n")
print(summary(fit))

cat("\n=== Predictions (population-level) ===\n")
newdata <- data.frame(
  Days = c(0, 1, 5, 10),
  Subject = factor(rep("308", 4), levels = levels(factor(data$Subject)))
)
preds <- predict(fit, newdata, re.form = NA)
cat("Predictions:\n")
print(preds)
