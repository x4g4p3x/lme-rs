source("comparisons/parity/parity_r_setup.R")
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})
source("comparisons/r_setup.R")

data <- read.csv("tests/data/sleepstudy.csv")
data$Subject <- as.factor(data$Subject)
fit <- lmer(
  Reaction ~ Days + offset(OffsetDays10) + (Days | Subject),
  data = data,
  REML = TRUE
)
coefs <- as.list(as.numeric(fixef(fit)))
names(coefs) <- names(fixef(fit))
cat(toJSON(
  list(case = "sleepstudy_offset", implementation = "r", coefficients = coefs),
  auto_unbox = TRUE,
  digits = 10
))
