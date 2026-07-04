source("comparisons/parity/parity_r_setup.R")
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})
source("comparisons/r_setup.R")

data <- read.csv("tests/data/cbpp_binary.csv")
data$herd <- as.factor(data$herd)
fit <- glmer(
  y ~ period2 + period3 + period4 + (1 | herd),
  data = data,
  family = binomial(link = "probit")
)
coefs <- as.list(as.numeric(fixef(fit)))
names(coefs) <- names(fixef(fit))
cat(toJSON(
  list(case = "glmm_probit", implementation = "r", coefficients = coefs),
  auto_unbox = TRUE,
  digits = 10
))
