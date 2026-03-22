suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(car))

cat("\n=== R (lme4 + lmerTest) ===\n")
data <- read.csv("tests/data/pastes.csv")
data$cask <- as.factor(data$cask)
data$batch <- as.factor(data$batch)

fit <- lmer(strength ~ cask + (1 | batch), data = data, REML = TRUE)
print(summary(fit))

cat("\n=== Type III ANOVA (Satterthwaite) ===\n")
print(anova(fit, ddf="Satterthwaite", type=3))

cat("\n=== Type III ANOVA (Wald Chi-Square approximation for joint) ===\n")
print(car::Anova(fit, type=3, test.statistic="Wald"))
