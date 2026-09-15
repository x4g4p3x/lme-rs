# Independent black-box reference outputs for synthetic project-owned data.
# Run from the repository root; requires car, emmeans, lmerTest, jsonlite.
# No third-party implementation source or datasets are incorporated.
suppressPackageStartupMessages({
  library(car)
  library(emmeans)
  library(lmerTest)
  library(jsonlite)
})

ordered_factors <- function(d) {
  d$a <- factor(d$a, levels = c("C", "A", "B"))
  d$b <- factor(d$b, levels = c("late", "early"))
  contrasts(d$a) <- contr.sum(3)
  contrasts(d$b) <- contr.sum(2)
  d
}

rows <- list()
for (a in 0:2) {
  for (b in 0:1) {
    for (j in seq_len(5 + a + 2 * b)) {
      x <- j / 3 + a / 4 - b / 5
      y <- 3 + a + 0.7 * b + 0.4 * a * b + 1.2 * x + sin(j * 1.7 + a + b)
      rows[[length(rows) + 1]] <- data.frame(a = c("C", "A", "B")[a + 1], b = c("late", "early")[b + 1], x = x, y = y)
    }
  }
}
d <- ordered_factors(do.call(rbind, rows))
m <- lm(terms(y ~ a + b + a:b + x, keep.order = TRUE), d)
table_rows <- function(tab, kind) {
  tab <- as.data.frame(tab)
  tab$term <- rownames(tab)
  names(tab)[names(tab) == "Sum Sq"] <- "ss"
  names(tab)[names(tab) == "Df"] <- "df"
  names(tab)[names(tab) %in% c("F value", "F")] <- "f"
  names(tab)[names(tab) == "Pr(>F)"] <- "p"
  tab <- tab[tab$term %in% c("a", "b", "a:b", "x"), c("term", "ss", "df", "f", "p")]
  list(type = kind, rows = tab)
}
at <- list(x = 0.75)
means <- as.data.frame(emmeans(m, ~ a | b, at = at))
pairs <- as.data.frame(pairs(emmeans(m, ~ a | b, at = at), reverse = TRUE, adjust = "holm"))
equal <- as.data.frame(emmeans(m, ~a, at = at, weights = "equal"))
proportional <- as.data.frame(emmeans(m, ~a, at = at, weights = "proportional"))
ols <- list(
  data = d, fitted = unname(fitted(m)),
  coefficients = unname(coef(m)[c("(Intercept)", "a1", "a2", "b1", "a1:b1", "a2:b1", "x")]),
  ci = unname(confint(m)[c("(Intercept)", "a1", "a2", "b1", "a1:b1", "a2:b1", "x"), ]),
  anova = list(table_rows(anova(m), "I"), table_rows(Anova(m, type = 2), "II"), table_rows(Anova(m, type = 3), "III")),
  residual_df = df.residual(m), residual_ss = sum(resid(m)^2),
  means = means, pairs = pairs, equal = equal, proportional = proportional
)

rows <- list()
for (i in 0:17) {
  for (t in 0:3) {
    if ((i %% 4 == 0 && t == 2) || (i %% 5 == 0 && t == 3)) next
    a <- i %% 3
    x <- t + 0.15 * (i %% 4)
    y <- 5 + 0.35 * a + 0.3 * t + 0.13 * a * t + 1.4 * sin(i * 1.37) + 0.5 * cos(i * 2.1 + t * 1.8)
    rows[[length(rows) + 1]] <- data.frame(id = paste0("s", i), a = c("C", "A", "B")[a + 1], x = x, y = y)
  }
}
l <- do.call(rbind, rows)
l$a <- factor(l$a, levels = c("C", "A", "B"))
contrasts(l$a) <- contr.sum(3)
full <- lmer(y ~ a * x + (1 | id), l, REML = FALSE)
null <- lmer(y ~ a + x + (1 | id), l, REML = FALSE)
reml <- update(full, REML = TRUE)
tab <- as.data.frame(anova(reml, type = 3, ddf = "Satterthwaite"))
tab$term <- rownames(tab)
names(tab) <- c("ss", "ms", "num_df", "den_df", "f", "p", "term")
set.seed(712)
ys <- simulate(null, nsim = 3, use.u = FALSE)
refits <- lapply(ys, function(y) {
  f <- refit(full, y)
  n <- refit(null, y)
  list(y = unname(y), full_deviance = -2 * as.numeric(logLik(f)), null_deviance = -2 * as.numeric(logLik(n)))
})
lmm <- list(
  data = l, fitted = unname(predict(full, re.form = NA)),
  full_deviance = -2 * as.numeric(logLik(full)), null_deviance = -2 * as.numeric(logLik(null)),
  anova = tab, refits = unname(refits)
)

out <- list(
  provenance = list(
    description = "Project-owned deterministic synthetic observations; black-box R reference calls",
    R = R.version.string, packages = sapply(c("car", "emmeans", "lmerTest", "lme4", "jsonlite"), function(p) as.character(packageVersion(p)))
  ),
  ols = ols, lmm = lmm
)
write_json(out, "tests/data/modeling_reference.json", auto_unbox = TRUE, pretty = TRUE, digits = 16, dataframe = "rows", na = "null")
