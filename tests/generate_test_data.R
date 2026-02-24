#!/usr/bin/env Rscript

# Create a local library directory to circumvent permission errors
dir.create("rlib", showWarnings = FALSE)
.libPaths(c("rlib", .libPaths()))

# Check and install lme4 and jsonlite if needed
if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4", repos = "https://cloud.r-project.org", lib = "rlib")
}
if (!requireNamespace("pbkrtest", quietly = TRUE)) {
  install.packages("pbkrtest", repos = "https://cloud.r-project.org", lib = "rlib")
}
if (!requireNamespace("lmerTest", quietly = TRUE)) {
  install.packages("lmerTest", repos = "https://cloud.r-project.org", lib = "rlib")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org", lib = "rlib")
}

library(lme4)
library(pbkrtest)
library(lmerTest)
library(jsonlite)

# Simple intercept-only model: Reaction ~ 1 + (1 | Subject) using sleepstudy
data("sleepstudy", package = "lme4")
# Models
fm1 <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy, REML = TRUE)
fm2 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = TRUE)

get_model_data <- function(model_str, fit) {
  X <- getME(fit, "X")
  Z <- getME(fit, "Z")
  Zt <- getME(fit, "Zt")
  y <- getME(fit, "y")
  theta <- getME(fit, "theta")
  beta <- fixef(fit)
  objective_val <- as.numeric(if (isREML(fit)) lme4::REMLcrit(fit) else deviance(fit))
  
  # Try to get Kenward-Roger dof and p-values (if it's an LMM)
  kr_dof <- NULL
  kr_p <- NULL
  if (is(fit, "lmerMod") || is(fit, "lmerModLmerTest")) {
      tryCatch({
          # We can use lmerTest's summary or anova to get KR
          # Let's get them from the summary using pbkrtest/lmerTest integration
          anova_res <- anova(as(fit, "lmerModLmerTest"), ddf = "Kenward-Roger")
          # These are F-tests, but we also want t-test DoFs for coefficients
          summary_res <- summary(as(fit, "lmerModLmerTest"), ddf = "Kenward-Roger")
          
          # Coefficient table has Estimate, Std. Error, df, t value, Pr(>|t|)
          coef_table <- coef(summary_res)
          
          if ("df" %in% colnames(coef_table)) {
              kr_dof <- as.numeric(coef_table[, "df"])
          }
          if ("Pr(>|t|)" %in% colnames(coef_table)) {
              kr_p <- as.numeric(coef_table[, "Pr(>|t|)"])
          }
      }, error = function(e) {
          message("Failed to compute KR dof for model: ", e$message)
      })
  }

  outputs <- list(
    theta = as.numeric(theta),
    beta = as.numeric(beta),
    reml_crit = unbox(objective_val)
  )
  
  if (!is.null(kr_dof)) {
      outputs$kr_dof <- kr_dof
  }
  if (!is.null(kr_p)) {
      outputs$kr_p <- kr_p
  }
  
  list(
    model = unbox(model_str),
    inputs = list(
      X = as.matrix(X),
      Zt = as.matrix(Zt),
      y = as.numeric(y)
    ),
    outputs = outputs
  )
}

out1 <- get_model_data("Reaction ~ 1 + (1 | Subject)", fm1)
out2 <- get_model_data("Reaction ~ Days + (Days | Subject)", fm2)

# Crossed random effects model
data("Penicillin", package = "lme4")
fm3 <- lmer(diameter ~ 1 + (1 | plate) + (1 | sample), data = Penicillin, REML = TRUE)
out3 <- get_model_data("diameter ~ 1 + (1 | plate) + (1 | sample)", fm3)

# ML Optimization baseline
fm4 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = FALSE)
out4 <- get_model_data("Reaction ~ Days + (Days | Subject) [ML]", fm4)

get_glmm_data <- function(model_str, fit) {
  X <- getME(fit, "X")
  Z <- getME(fit, "Z")
  Zt <- getME(fit, "Zt")
  y <- getME(fit, "y")
  theta <- getME(fit, "theta")
  beta <- fixef(fit)
  dev_val <- deviance(fit)
  
  list(
    model = unbox(model_str),
    inputs = list(
      X = as.matrix(X),
      Zt = as.matrix(Zt),
      y = as.numeric(y)
    ),
    outputs = list(
      theta = as.numeric(theta),
      beta = as.numeric(beta),
      deviance = unbox(dev_val)
    )
  )
}

# GLMM 1: Binomial (cbpp, but modify the response to a single vector of 1s and 0s or probabilities for easier testing on Rust side)
data("cbpp", package = "lme4")
cbpp_binary <- data.frame(herd = character(), period2 = numeric(), period3 = numeric(), period4 = numeric(), incidence = numeric())
for (i in 1:nrow(cbpp)) {
  s <- cbpp$size[i]
  inc <- cbpp$incidence[i]
  if (s > 0) {
    herd_rep <- rep(as.character(cbpp$herd[i]), s)
    period_val <- as.numeric(cbpp$period[i])
    p2 <- rep(ifelse(period_val == 2, 1, 0), s)
    p3 <- rep(ifelse(period_val == 3, 1, 0), s)
    p4 <- rep(ifelse(period_val == 4, 1, 0), s)
    y_rep <- c(rep(1, inc), rep(0, s - inc))
    cbpp_binary <- rbind(cbpp_binary, data.frame(herd=herd_rep, period2=p2, period3=p3, period4=p4, y=y_rep))
  }
}
fm_binom <- glmer(y ~ period2 + period3 + period4 + (1 | herd), data = cbpp_binary, family = binomial)
out_binom <- get_glmm_data("y ~ period2 + period3 + period4 + (1 | herd) [Binomial]", fm_binom)

# GLMM 2: Poisson (grouseticks)
data("grouseticks", package = "lme4")
# YEAR has 3 levels: 95, 96, 97. Make numeric dummies.
grouseticks$YEAR96 <- ifelse(grouseticks$YEAR == 96, 1, 0)
grouseticks$YEAR97 <- ifelse(grouseticks$YEAR == 97, 1, 0)
fm_pois <- glmer(TICKS ~ YEAR96 + YEAR97 + (1 | BROOD), data = grouseticks, family = poisson)
out_pois <- get_glmm_data("TICKS ~ YEAR96 + YEAR97 + (1 | BROOD) [Poisson]", fm_pois)

# Save to JSON
dir.create("tests/data", recursive = TRUE, showWarnings = FALSE)
write_json(out1, "tests/data/intercept_only.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out2, "tests/data/random_slopes.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out3, "tests/data/penicillin.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out4, "tests/data/mock_ml.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_binom, "tests/data/glmm_binomial.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_pois, "tests/data/glmm_poisson.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write.csv(sleepstudy, "tests/data/sleepstudy.csv", row.names = FALSE)
write.csv(cbpp_binary, "tests/data/cbpp_binary.csv", row.names = FALSE)
write.csv(grouseticks, "tests/data/grouseticks.csv", row.names = FALSE)
cat("Successfully generated test data.\n")
