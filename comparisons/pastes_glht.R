#!/usr/bin/env Rscript
# Wald pairwise MCP for pastes `cask` (lme4 vcov + stats::ptukey).
# Does not require multcomp; Tukey-Kramer uses df = Inf like glht() on merMod.

suppressPackageStartupMessages(library(lme4))

df <- read.csv("tests/data/pastes.csv")
df$cask <- factor(df$cask)
m <- lmer(strength ~ cask + (1 | batch), df, REML = TRUE)
b <- fixef(m)
V <- as.matrix(vcov(m))
L <- rbind(
  `b - a` = c(0, 1, 0),
  `c - a` = c(0, 0, 1),
  `c - b` = c(0, -1, 1)
)
est <- as.numeric(L %*% b)
se <- sqrt(diag(L %*% V %*% t(L)))
z <- est / se
p_raw <- 2 * pnorm(-abs(z))
p_bonf <- p.adjust(p_raw, "bonferroni")
p_holm <- p.adjust(p_raw, "holm")
p_tukey <- 1 - ptukey(sqrt(2) * abs(z), nmeans = 3, df = Inf)

out <- data.frame(
  contrast = rownames(L),
  estimate = est,
  se = se,
  z = z,
  p_raw = p_raw,
  p_bonferroni = p_bonf,
  p_holm = p_holm,
  p_tukey = p_tukey
)
print(out, digits = 16)
cat("\nWrite tests/data/pastes_glht_tukey.json after copying these values.\n")
