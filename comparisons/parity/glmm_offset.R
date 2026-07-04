source("comparisons/parity/parity_r_setup.R")
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})
source("comparisons/r_setup.R")

data <- read.csv("tests/data/grouseticks.csv")
data$BROOD <- as.factor(data$BROOD)
fit <- glmer(
  TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD),
  data = data,
  family = poisson(link = "log")
)
coefs <- as.list(as.numeric(fixef(fit)))
names(coefs) <- names(fixef(fit))
cat(toJSON(
  list(case = "glmm_offset", implementation = "r", coefficients = coefs),
  auto_unbox = TRUE,
  digits = 10
))
