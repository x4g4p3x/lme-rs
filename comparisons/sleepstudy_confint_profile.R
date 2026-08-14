#!/usr/bin/env Rscript
# lme4 profile CIs for sleepstudy random-intercept ML, including variance components.
# Regenerates the numbers stored in tests/data/sleepstudy_confint_profile.json
# and tests/data/sleepstudy_confint_profile_vc.json.

library(lme4)

sleep <- read.csv("tests/data/sleepstudy.csv")
fm <- lmer(Reaction ~ Days + (1 | Subject), data = sleep, REML = FALSE)

cat("theta:", paste(getME(fm, "theta"), collapse = ", "), "\n")
cat("sigma:", sigma(fm), "\n")
cat("sig01 (theta * sigma):", getME(fm, "theta")[1] * sigma(fm), "\n")

ci <- confint(fm, method = "profile", level = 0.95, oldNames = TRUE, quiet = TRUE)
print(ci)
