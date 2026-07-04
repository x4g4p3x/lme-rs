#!/usr/bin/env Rscript

# Create a local library directory to circumvent permission errors.
dir.create("rlib", showWarnings = FALSE)
.libPaths(c("rlib", .libPaths()))

required_packages <- c("lme4", "pbkrtest", "lmerTest", "jsonlite", "clubSandwich")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", lib = "rlib")
  }
}

library(lme4)
library(pbkrtest)
library(lmerTest)
library(jsonlite)
library(clubSandwich)

pkg_version <- function(pkg) {
  jsonlite::unbox(as.character(utils::packageVersion(pkg)))
}

scalar_check <- function(name, value, tolerance) {
  list(
    name = jsonlite::unbox(name),
    value = jsonlite::unbox(as.numeric(value)),
    tolerance = jsonlite::unbox(as.numeric(tolerance))
  )
}

prediction_check <- function(name, newdata, values, tolerance, allow_new_levels = FALSE) {
  list(
    name = jsonlite::unbox(name),
    newdata = newdata,
    values = as.numeric(values),
    tolerance = jsonlite::unbox(as.numeric(tolerance)),
    allow_new_levels = jsonlite::unbox(allow_new_levels)
  )
}

anova_check <- function(term, f_value, p_value, num_df, den_df, f_tol = 0.01, p_tol = 0.000002, df_tol = 0.01) {
  list(
    term = jsonlite::unbox(term),
    f_value = scalar_check("F", f_value, f_tol),
    p_value = scalar_check("p", p_value, p_tol),
    num_df = scalar_check("NumDF", num_df, 0.000001),
    den_df = scalar_check("DenDF", den_df, df_tol)
  )
}

# Simple intercept-only model: Reaction ~ 1 + (1 | Subject) using sleepstudy.
data("sleepstudy", package = "lme4")

# Models.
fm1 <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy, REML = TRUE)
fm2 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = TRUE)

get_model_data <- function(model_str, fit) {
  X <- getME(fit, "X")
  Zt <- getME(fit, "Zt")
  y <- getME(fit, "y")
  theta <- getME(fit, "theta")
  beta <- fixef(fit)
  objective_val <- as.numeric(if (isREML(fit)) lme4::REMLcrit(fit) else deviance(fit))
  coef_table_default <- coef(summary(fit))
  beta_se <- as.numeric(coef_table_default[, "Std. Error"])

  # Try to get Kenward-Roger dof and p-values (if it's an LMM).
  kr_dof <- NULL
  kr_p <- NULL
  kr_anova_ndf <- NULL
  kr_anova_ddf <- NULL
  kr_anova_f <- NULL
  kr_anova_p <- NULL

  sat_anova_ndf <- NULL
  sat_anova_ddf <- NULL
  sat_anova_f <- NULL
  sat_anova_p <- NULL

  if (is(fit, "lmerMod") || is(fit, "lmerModLmerTest")) {
    tryCatch({
      # We can use lmerTest's summary or anova to get KR and Satterthwaite.
      anova_res_kr <- anova(as(fit, "lmerModLmerTest"), ddf = "Kenward-Roger", type = 3)
      anova_res_sat <- anova(as(fit, "lmerModLmerTest"), ddf = "Satterthwaite", type = 3)
      summary_res <- summary(as(fit, "lmerModLmerTest"), ddf = "Kenward-Roger")

      # Coefficient table has Estimate, Std. Error, df, t value, Pr(>|t|).
      coef_table <- coef(summary_res)

      if ("df" %in% colnames(coef_table)) {
        kr_dof <- as.numeric(coef_table[, "df"])
      }
      if ("Pr(>|t|)" %in% colnames(coef_table)) {
        kr_p <- as.numeric(coef_table[, "Pr(>|t|)"])
      }

      # Anova table has NumDF, DenDF, F value, Pr(>F).
      if ("NumDF" %in% colnames(anova_res_kr)) {
        kr_anova_ndf <- as.numeric(anova_res_kr[, "NumDF"])
      }
      if ("DenDF" %in% colnames(anova_res_kr)) {
        kr_anova_ddf <- as.numeric(anova_res_kr[, "DenDF"])
      }
      if ("F value" %in% colnames(anova_res_kr)) {
        kr_anova_f <- as.numeric(anova_res_kr[, "F value"])
      }
      if ("Pr(>F)" %in% colnames(anova_res_kr)) {
        kr_anova_p <- as.numeric(anova_res_kr[, "Pr(>F)"])
      }

      # Satterthwaite Anova.
      if ("NumDF" %in% colnames(anova_res_sat)) {
        sat_anova_ndf <- as.numeric(anova_res_sat[, "NumDF"])
      }
      if ("DenDF" %in% colnames(anova_res_sat)) {
        sat_anova_ddf <- as.numeric(anova_res_sat[, "DenDF"])
      }
      if ("F value" %in% colnames(anova_res_sat)) {
        sat_anova_f <- as.numeric(anova_res_sat[, "F value"])
      }
      if ("Pr(>F)" %in% colnames(anova_res_sat)) {
        sat_anova_p <- as.numeric(anova_res_sat[, "Pr(>F)"])
      }
    }, error = function(e) {
      message("Failed to compute KR/Satterthwaite outputs for model: ", e$message)
    })
  }

  outputs <- list(
    theta = as.numeric(theta),
    beta = as.numeric(beta),
    beta_se = beta_se,
    sigma2 = jsonlite::unbox(as.numeric(sigma(fit)^2)),
    reml_crit = jsonlite::unbox(objective_val)
  )

  if (!is.null(kr_dof)) {
    outputs$kr_dof <- kr_dof
  }
  if (!is.null(kr_p)) {
    outputs$kr_p <- kr_p
  }
  if (!is.null(kr_anova_f)) {
    outputs$kr_anova_f <- kr_anova_f
    outputs$kr_anova_p <- kr_anova_p
    outputs$kr_anova_ndf <- kr_anova_ndf
    outputs$kr_anova_ddf <- kr_anova_ddf
  }
  if (!is.null(sat_anova_f)) {
    outputs$sat_anova_f <- sat_anova_f
    outputs$sat_anova_p <- sat_anova_p
    outputs$sat_anova_ndf <- sat_anova_ndf
    outputs$sat_anova_ddf <- sat_anova_ddf
  }

  # Robust Standard Errors (CR0).
  tryCatch({
    vcov_cr0 <- vcovCR(fit, type = "CR0")
    v_beta_robust <- as.matrix(vcov_cr0)
    robust_se <- sqrt(diag(v_beta_robust))
    robust_t <- as.numeric(beta) / robust_se

    outputs$robust_v_beta <- as.vector(v_beta_robust)
    outputs$robust_se <- robust_se
    outputs$robust_t <- robust_t
  }, error = function(e) {
    message("Skipping robust SEs: ", e$message)
  })

  list(
    model = jsonlite::unbox(model_str),
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

# Crossed random effects model.
data("Penicillin", package = "lme4")
fm3 <- lmer(diameter ~ 1 + (1 | plate) + (1 | sample), data = Penicillin, REML = TRUE)
out3 <- get_model_data("diameter ~ 1 + (1 | plate) + (1 | sample)", fm3)

# Pastes: categorical fixed effect with multi-DoF Type III ANOVA (Satterthwaite).
pastes_data <- read.csv("tests/data/pastes.csv")
pastes_data$cask <- as.factor(pastes_data$cask)
pastes_data$batch <- as.factor(pastes_data$batch)
fm_pastes_cask <- lmer(strength ~ cask + (1 | batch), data = pastes_data, REML = TRUE)
out_pastes_cask <- get_model_data("strength ~ cask + (1 | batch)", fm_pastes_cask)

# ML Optimization baseline.
fm4 <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = FALSE)
out4 <- get_model_data("Reaction ~ Days + (Days | Subject) [ML]", fm4)

get_glmm_data <- function(model_str, fit) {
  X <- getME(fit, "X")
  Zt <- getME(fit, "Zt")
  y <- getME(fit, "y")
  theta <- getME(fit, "theta")
  beta <- fixef(fit)
  dev_val <- deviance(fit)

  list(
    model = jsonlite::unbox(model_str),
    inputs = list(
      X = as.matrix(X),
      Zt = as.matrix(Zt),
      y = as.numeric(y)
    ),
    outputs = list(
      theta = as.numeric(theta),
      beta = as.numeric(beta),
      deviance = jsonlite::unbox(dev_val)
    )
  )
}

# GLMM 1: Binomial (CBPP expanded from grouped binomial counts to binary trials for Rust-side testing).
data("cbpp", package = "lme4")
cbpp_binary <- data.frame(
  herd = character(),
  period2 = numeric(),
  period3 = numeric(),
  period4 = numeric(),
  incidence = numeric()
)
for (i in seq_len(nrow(cbpp))) {
  s <- cbpp$size[i]
  inc <- cbpp$incidence[i]
  if (s > 0) {
    herd_rep <- rep(as.character(cbpp$herd[i]), s)
    period_val <- as.numeric(cbpp$period[i])
    p2 <- rep(ifelse(period_val == 2, 1, 0), s)
    p3 <- rep(ifelse(period_val == 3, 1, 0), s)
    p4 <- rep(ifelse(period_val == 4, 1, 0), s)
    y_rep <- c(rep(1, inc), rep(0, s - inc))
    cbpp_binary <- rbind(cbpp_binary, data.frame(herd = herd_rep, period2 = p2, period3 = p3, period4 = p4, y = y_rep))
  }
}
fm_binom <- glmer(y ~ period2 + period3 + period4 + (1 | herd), data = cbpp_binary, family = binomial)
out_binom <- get_glmm_data("y ~ period2 + period3 + period4 + (1 | herd) [Binomial]", fm_binom)

# GLMM 2: Poisson (grouseticks).
data("grouseticks", package = "lme4")
# YEAR has 3 levels: 95, 96, 97. Make numeric dummies.
grouseticks$YEAR96 <- ifelse(grouseticks$YEAR == 96, 1, 0)
grouseticks$YEAR97 <- ifelse(grouseticks$YEAR == 97, 1, 0)
fm_pois <- glmer(TICKS ~ YEAR96 + YEAR97 + (1 | BROOD), data = grouseticks, family = poisson)
out_pois <- get_glmm_data("TICKS ~ YEAR96 + YEAR97 + (1 | BROOD) [Poisson]", fm_pois)

# LMM with offset (sleepstudy): offset column must exist in CSV.
sleepstudy$OffsetDays10 <- sleepstudy$Days * 10
fm_offset <- lmer(
  Reaction ~ Days + offset(OffsetDays10) + (Days | Subject),
  data = sleepstudy,
  REML = TRUE
)
out_offset <- get_model_data(
  "Reaction ~ Days + offset(OffsetDays10) + (Days | Subject)",
  fm_offset
)

# GLMM with non-canonical link (probit CBPP).
fm_binom_probit <- glmer(
  y ~ period2 + period3 + period4 + (1 | herd),
  data = cbpp_binary,
  family = binomial(link = "probit")
)
out_binom_probit <- get_glmm_data(
  "y ~ period2 + period3 + period4 + (1 | herd) [Binomial probit]",
  fm_binom_probit
)

# GLMM with prior observation weights (reproducible).
set.seed(42)
cbpp_binary$prior_w <- runif(nrow(cbpp_binary), 0.5, 1.5)
fm_binom_wt <- glmer(
  y ~ period2 + period3 + period4 + (1 | herd),
  data = cbpp_binary,
  family = binomial,
  weights = prior_w
)
out_binom_wt <- get_glmm_data(
  "y ~ period2 + period3 + period4 + (1 | herd) [Binomial weighted]",
  fm_binom_wt
)

# Poisson GLMM with offset on log(HEIGHT).
grouseticks$log_height <- log(grouseticks$HEIGHT)
fm_pois_off <- glmer(
  TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD),
  data = grouseticks,
  family = poisson
)
out_pois_off <- get_glmm_data(
  "TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD) [Poisson]",
  fm_pois_off
)

# Nonlinear mixed model (Orange / SSlogis).
data("Orange", package = "datasets")
Orange$Tree <- as.factor(Orange$Tree)
fm_orange <- nlmer(
  circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym | Tree,
  data = Orange,
  start = c(Asym = 200, xmid = 725, scal = 350)
)
orange_fix <- fixef(fm_orange)
orange_vc_sd <- as.numeric(attr(VarCorr(fm_orange)$Tree, "stddev")["Asym"])
orange_pop_ages <- c(118, 484, 664)
orange_pop_circ <- as.numeric(
  SSlogis(orange_pop_ages, orange_fix["Asym"], orange_fix["xmid"], orange_fix["scal"])
)
orange_cond_pred <- fitted(fm_orange)[Orange$Tree == "1"][1:3]
out_orange <- list(
  model = jsonlite::unbox("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree"),
  outputs = list(
    beta = as.numeric(fixef(fm_orange)),
    theta = as.numeric(getME(fm_orange, "theta")),
    re_sd = jsonlite::unbox(orange_vc_sd),
    sigma2 = jsonlite::unbox(as.numeric(sigma(fm_orange))^2),
    logLik = jsonlite::unbox(as.numeric(logLik(fm_orange))),
    pop_pred = orange_pop_circ,
    cond_pred = as.numeric(orange_cond_pred)
  )
)

sleepstudy_pop_newdata <- data.frame(Days = c(0, 1, 5, 10), Subject = factor(rep("308", 4), levels = levels(sleepstudy$Subject)))
sleepstudy_cond_newdata <- data.frame(Days = c(0, 1, 5), Subject = factor(rep("308", 3), levels = levels(sleepstudy$Subject)))

sleepstudy_pop_pred <- predict(fm2, newdata = sleepstudy_pop_newdata, re.form = NA)
sleepstudy_cond_pred <- predict(fm2, newdata = sleepstudy_cond_newdata, re.form = NULL, allow.new.levels = FALSE)

build_golden_manifest <- function() {
  list(
    schema_version = jsonlite::unbox(1),
    reference_environment = list(
      engine = jsonlite::unbox("R"),
      generator = jsonlite::unbox("tests/generate_test_data.R"),
      r_version = jsonlite::unbox(R.version.string),
      platform = jsonlite::unbox(R.version$platform),
      notes = c(
        "Reference values are generated from R mixed-model tooling and stored with per-check tolerances.",
        "GLMM deviance/log-likelihood levels are intentionally not asserted because lme-rs and lme4 use different data-dependent constants. Coefficients and variance parameters are the parity targets for GLMM cases.",
        "Tolerance choices are part of the golden fixture because optimizer and approximation differences are expected for some models."
      ),
      packages = list(
        lme4 = pkg_version("lme4"),
        lmerTest = pkg_version("lmerTest"),
        pbkrtest = pkg_version("pbkrtest"),
        clubSandwich = pkg_version("clubSandwich"),
        jsonlite = pkg_version("jsonlite")
      )
    ),
    cases = list(
      list(
        id = jsonlite::unbox("sleepstudy_random_slopes_reml"),
        description = jsonlite::unbox("Canonical lme4 sleepstudy random-intercept/random-slope LMM fitted by REML."),
        kind = jsonlite::unbox("lmm"),
        data_path = jsonlite::unbox("tests/data/sleepstudy.csv"),
        formula = jsonlite::unbox("Reaction ~ Days + (Days | Subject)"),
        reml = jsonlite::unbox(TRUE),
        reference = list(
          engine = jsonlite::unbox("lme4::lmer"),
          call = jsonlite::unbox("lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = TRUE)"),
          source_fixture = jsonlite::unbox("tests/data/random_slopes.json")
        ),
        post_fit = list(
          robust_cluster = jsonlite::unbox("Subject"),
          satterthwaite = jsonlite::unbox(TRUE),
          kenward_roger = jsonlite::unbox(TRUE)
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out2$outputs$beta[1], 0.05),
            scalar_check("Days", out2$outputs$beta[2], 0.05)
          ),
          theta = list(
            scalar_check("Subject.(Intercept)", out2$outputs$theta[1], 0.02),
            scalar_check("Subject.Days.(Intercept)", out2$outputs$theta[2], 0.02),
            scalar_check("Subject.Days", out2$outputs$theta[3], 0.02)
          ),
          sigma2 = scalar_check("sigma2", out2$outputs$sigma2, 0.1),
          deviance = scalar_check("REML criterion", out2$outputs$reml_crit, 0.1),
          beta_se = list(
            scalar_check("SE((Intercept))", out2$outputs$beta_se[1], 0.02),
            scalar_check("SE(Days)", out2$outputs$beta_se[2], 0.02)
          ),
          population_predictions = list(
            prediction_check(
              "predict(re.form = NA) for Days 0,1,5,10",
              list(Days = c(0, 1, 5, 10), Subject = c("308", "308", "308", "308")),
              sleepstudy_pop_pred,
              0.000001
            )
          ),
          conditional_predictions = list(
            prediction_check(
              "predict for Subject 308 including conditional modes",
              list(Days = c(0, 1, 5), Subject = c("308", "308", "308")),
              sleepstudy_cond_pred,
              0.1,
              allow_new_levels = FALSE
            )
          ),
          satterthwaite_anova = list(
            anova_check("Days", out2$outputs$sat_anova_f[1], out2$outputs$sat_anova_p[1], out2$outputs$sat_anova_ndf[1], out2$outputs$sat_anova_ddf[1])
          ),
          kenward_roger_anova = list(
            anova_check("Days", out2$outputs$kr_anova_f[1], out2$outputs$kr_anova_p[1], out2$outputs$kr_anova_ndf[1], out2$outputs$kr_anova_ddf[1])
          ),
          robust_se = list(
            scalar_check("CR0 SE((Intercept))", out2$outputs$robust_se[1], 0.0001),
            scalar_check("CR0 SE(Days)", out2$outputs$robust_se[2], 0.0001)
          ),
          robust_t = list(
            scalar_check("CR0 t((Intercept))", out2$outputs$robust_t[1], 0.0001),
            scalar_check("CR0 t(Days)", out2$outputs$robust_t[2], 0.0001)
          )
        )
      ),
      list(
        id = jsonlite::unbox("sleepstudy_random_slopes_ml"),
        description = jsonlite::unbox("Same sleepstudy random-slope model fitted by ML for model-comparison workflows."),
        kind = jsonlite::unbox("lmm"),
        data_path = jsonlite::unbox("tests/data/sleepstudy.csv"),
        formula = jsonlite::unbox("Reaction ~ Days + (Days | Subject)"),
        reml = jsonlite::unbox(FALSE),
        reference = list(
          engine = jsonlite::unbox("lme4::lmer"),
          call = jsonlite::unbox("lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = FALSE)"),
          source_fixture = jsonlite::unbox("tests/data/mock_ml.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out4$outputs$beta[1], 0.05),
            scalar_check("Days", out4$outputs$beta[2], 0.05)
          ),
          theta = list(
            scalar_check("Subject.(Intercept)", out4$outputs$theta[1], 0.02),
            scalar_check("Subject.Days.(Intercept)", out4$outputs$theta[2], 0.02),
            scalar_check("Subject.Days", out4$outputs$theta[3], 0.02)
          ),
          deviance = scalar_check("ML deviance", out4$outputs$reml_crit, 0.1)
        )
      ),
      list(
        id = jsonlite::unbox("pastes_cask_multi_dof_reml"),
        description = jsonlite::unbox("Pastes strength ~ cask with batch RE; multi-DoF Type III Satterthwaite ANOVA for cask."),
        kind = jsonlite::unbox("lmm"),
        data_path = jsonlite::unbox("tests/data/pastes.csv"),
        formula = jsonlite::unbox("strength ~ cask + (1 | batch)"),
        reml = jsonlite::unbox(TRUE),
        reference = list(
          engine = jsonlite::unbox("lme4::lmer"),
          call = jsonlite::unbox("lmer(strength ~ cask + (1 | batch), pastes, REML = TRUE)"),
          source_fixture = jsonlite::unbox("tests/data/pastes_cask_reml.json")
        ),
        post_fit = list(
          satterthwaite = jsonlite::unbox(TRUE)
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_pastes_cask$outputs$beta[1], 0.05),
            scalar_check("caskb", out_pastes_cask$outputs$beta[2], 0.05),
            scalar_check("caskc", out_pastes_cask$outputs$beta[3], 0.05)
          ),
          theta = list(
            scalar_check("batch.(Intercept)", out_pastes_cask$outputs$theta[1], 0.02)
          ),
          sigma2 = scalar_check("sigma2", out_pastes_cask$outputs$sigma2, 0.1),
          deviance = scalar_check("REML criterion", out_pastes_cask$outputs$reml_crit, 0.1),
          satterthwaite_anova = list(
            anova_check(
              "cask",
              out_pastes_cask$outputs$sat_anova_f[1],
              out_pastes_cask$outputs$sat_anova_p[1],
              out_pastes_cask$outputs$sat_anova_ndf[1],
              out_pastes_cask$outputs$sat_anova_ddf[1],
              p_tol = 0.0001
            )
          )
        )
      ),
      list(
        id = jsonlite::unbox("penicillin_crossed_reml"),
        description = jsonlite::unbox("Crossed random-intercept LMM from lme4 Penicillin data."),
        kind = jsonlite::unbox("lmm"),
        data_path = jsonlite::unbox("tests/data/penicillin.csv"),
        formula = jsonlite::unbox("diameter ~ 1 + (1 | plate) + (1 | sample)"),
        reml = jsonlite::unbox(TRUE),
        reference = list(
          engine = jsonlite::unbox("lme4::lmer"),
          call = jsonlite::unbox("lmer(diameter ~ 1 + (1 | plate) + (1 | sample), Penicillin, REML = TRUE)"),
          source_fixture = jsonlite::unbox("tests/data/penicillin.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out3$outputs$beta[1], 0.05)
          ),
          theta = list(
            scalar_check("plate.(Intercept)", out3$outputs$theta[1], 0.05),
            scalar_check("sample.(Intercept)", out3$outputs$theta[2], 0.05)
          ),
          deviance = scalar_check("REML criterion", out3$outputs$reml_crit, 0.2)
        )
      ),
      list(
        id = jsonlite::unbox("cbpp_binomial_laplace"),
        description = jsonlite::unbox("Binomial GLMM using expanded CBPP binary-trial data and Laplace approximation."),
        kind = jsonlite::unbox("glmm"),
        data_path = jsonlite::unbox("tests/data/cbpp_binary.csv"),
        formula = jsonlite::unbox("y ~ period2 + period3 + period4 + (1 | herd)"),
        family = jsonlite::unbox("binomial"),
        n_agq = jsonlite::unbox(1),
        reference = list(
          engine = jsonlite::unbox("lme4::glmer"),
          call = jsonlite::unbox("glmer(y ~ period2 + period3 + period4 + (1 | herd), cbpp_binary, family = binomial)"),
          source_fixture = jsonlite::unbox("tests/data/glmm_binomial.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_binom$outputs$beta[1], 0.05),
            scalar_check("period2", out_binom$outputs$beta[2], 0.05),
            scalar_check("period3", out_binom$outputs$beta[3], 0.05),
            scalar_check("period4", out_binom$outputs$beta[4], 0.05)
          ),
          theta = list(
            scalar_check("herd.(Intercept)", out_binom$outputs$theta[1], 0.05)
          ),
          finite_deviance = jsonlite::unbox(TRUE)
        )
      ),
      list(
        id = jsonlite::unbox("grouseticks_poisson_laplace"),
        description = jsonlite::unbox("Poisson GLMM for grouseticks with brood random intercept."),
        kind = jsonlite::unbox("glmm"),
        data_path = jsonlite::unbox("tests/data/grouseticks.csv"),
        formula = jsonlite::unbox("TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)"),
        family = jsonlite::unbox("poisson"),
        n_agq = jsonlite::unbox(1),
        reference = list(
          engine = jsonlite::unbox("lme4::glmer"),
          call = jsonlite::unbox("glmer(TICKS ~ YEAR96 + YEAR97 + (1 | BROOD), grouseticks, family = poisson)"),
          source_fixture = jsonlite::unbox("tests/data/glmm_poisson.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_pois$outputs$beta[1], 0.15),
            scalar_check("YEAR96", out_pois$outputs$beta[2], 0.15),
            scalar_check("YEAR97", out_pois$outputs$beta[3], 0.15)
          ),
          theta = list(
            scalar_check("BROOD.(Intercept)", out_pois$outputs$theta[1], 0.01)
          ),
          finite_deviance = jsonlite::unbox(TRUE)
        )
      ),
      list(
        id = jsonlite::unbox("sleepstudy_offset_reml"),
        description = jsonlite::unbox("Sleepstudy LMM with Days + offset(OffsetDays10) random slopes."),
        kind = jsonlite::unbox("lmm"),
        data_path = jsonlite::unbox("tests/data/sleepstudy.csv"),
        formula = jsonlite::unbox("Reaction ~ Days + offset(OffsetDays10) + (Days | Subject)"),
        reml = jsonlite::unbox(TRUE),
        reference = list(
          engine = jsonlite::unbox("lme4::lmer"),
          call = jsonlite::unbox("lmer(Reaction ~ Days + offset(OffsetDays10) + (Days | Subject), sleepstudy, REML = TRUE)"),
          source_fixture = jsonlite::unbox("tests/data/sleepstudy_offset_reml.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_offset$outputs$beta[1], 0.05),
            scalar_check("Days", out_offset$outputs$beta[2], 0.05)
          ),
          theta = list(
            scalar_check("Subject.(Intercept)", out_offset$outputs$theta[1], 0.02),
            scalar_check("Subject.Days.(Intercept)", out_offset$outputs$theta[2], 0.02),
            scalar_check("Subject.Days", out_offset$outputs$theta[3], 0.02)
          ),
          sigma2 = scalar_check("sigma2", out_offset$outputs$sigma2, 0.1)
        )
      ),
      list(
        id = jsonlite::unbox("cbpp_binomial_probit"),
        description = jsonlite::unbox("Binomial GLMM on CBPP with probit link."),
        kind = jsonlite::unbox("glmm"),
        data_path = jsonlite::unbox("tests/data/cbpp_binary.csv"),
        formula = jsonlite::unbox("y ~ period2 + period3 + period4 + (1 | herd)"),
        family = jsonlite::unbox("binomial"),
        link = jsonlite::unbox("probit"),
        n_agq = jsonlite::unbox(1),
        reference = list(
          engine = jsonlite::unbox("lme4::glmer"),
          call = jsonlite::unbox("glmer(..., family = binomial(link = 'probit'))"),
          source_fixture = jsonlite::unbox("tests/data/glmm_binomial_probit.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_binom_probit$outputs$beta[1], 0.08),
            scalar_check("period2", out_binom_probit$outputs$beta[2], 0.08),
            scalar_check("period3", out_binom_probit$outputs$beta[3], 0.08),
            scalar_check("period4", out_binom_probit$outputs$beta[4], 0.08)
          ),
          theta = list(
            scalar_check("herd.(Intercept)", out_binom_probit$outputs$theta[1], 0.08)
          ),
          finite_deviance = jsonlite::unbox(TRUE)
        )
      ),
      list(
        id = jsonlite::unbox("cbpp_binomial_weighted"),
        description = jsonlite::unbox("Binomial GLMM on CBPP with prior observation weights."),
        kind = jsonlite::unbox("glmm"),
        data_path = jsonlite::unbox("tests/data/cbpp_binary_weighted.csv"),
        formula = jsonlite::unbox("y ~ period2 + period3 + period4 + (1 | herd)"),
        family = jsonlite::unbox("binomial"),
        weights_column = jsonlite::unbox("prior_w"),
        n_agq = jsonlite::unbox(1),
        reference = list(
          engine = jsonlite::unbox("lme4::glmer"),
          call = jsonlite::unbox("glmer(..., weights = prior_w)"),
          source_fixture = jsonlite::unbox("tests/data/glmm_binomial_weighted.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_binom_wt$outputs$beta[1], 0.08),
            scalar_check("period2", out_binom_wt$outputs$beta[2], 0.08),
            scalar_check("period3", out_binom_wt$outputs$beta[3], 0.08),
            scalar_check("period4", out_binom_wt$outputs$beta[4], 0.08)
          ),
          theta = list(
            scalar_check("herd.(Intercept)", out_binom_wt$outputs$theta[1], 0.08)
          ),
          finite_deviance = jsonlite::unbox(TRUE)
        )
      ),
      list(
        id = jsonlite::unbox("grouseticks_poisson_offset"),
        description = jsonlite::unbox("Poisson GLMM on grouseticks with offset(log_height)."),
        kind = jsonlite::unbox("glmm"),
        data_path = jsonlite::unbox("tests/data/grouseticks.csv"),
        formula = jsonlite::unbox("TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD)"),
        family = jsonlite::unbox("poisson"),
        n_agq = jsonlite::unbox(1),
        reference = list(
          engine = jsonlite::unbox("lme4::glmer"),
          call = jsonlite::unbox("glmer(TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD), ...)"),
          source_fixture = jsonlite::unbox("tests/data/glmm_poisson_offset.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("(Intercept)", out_pois_off$outputs$beta[1], 0.15),
            scalar_check("YEAR96", out_pois_off$outputs$beta[2], 0.15),
            scalar_check("YEAR97", out_pois_off$outputs$beta[3], 0.15)
          ),
          theta = list(
            scalar_check("BROOD.(Intercept)", out_pois_off$outputs$theta[1], 0.02)
          ),
          finite_deviance = jsonlite::unbox(TRUE)
        )
      ),
      list(
        id = jsonlite::unbox("orange_nlmer_sslogis"),
        description = jsonlite::unbox("Orange-tree nlmer with SSlogis mean and RE on Asym."),
        kind = jsonlite::unbox("nlmm"),
        data_path = jsonlite::unbox("tests/data/orange.csv"),
        formula = jsonlite::unbox("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree"),
        nlmm_start = list(
          Asym = jsonlite::unbox(200),
          xmid = jsonlite::unbox(725),
          scal = jsonlite::unbox(350)
        ),
        nlmm_reml = jsonlite::unbox(FALSE),
        reference = list(
          engine = jsonlite::unbox("lme4::nlmer"),
          call = jsonlite::unbox("nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree, Orange, start=c(Asym=200,xmid=725,scal=350))"),
          source_fixture = jsonlite::unbox("tests/data/orange_nlmer.json")
        ),
        expected = list(
          coefficients = list(
            scalar_check("Asym", out_orange$outputs$beta[1], 2.0),
            scalar_check("xmid", out_orange$outputs$beta[2], 2.0),
            scalar_check("scal", out_orange$outputs$beta[3], 2.0)
          ),
          theta = list(
            scalar_check("Tree.Asym", orange_vc_sd, 2.0)
          ),
          population_predictions = list(
            prediction_check(
              "population SSlogis circumference for Tree 1 ages 118,484,664",
              list(age = orange_pop_ages, Tree = c("1", "1", "1")),
              orange_pop_circ,
              1.0
            )
          ),
          conditional_predictions = list(
            prediction_check(
              "fitted values for Tree 1 (first 3 rows)",
              list(age = c(118, 484, 664), Tree = c("1", "1", "1")),
              orange_cond_pred,
              1.0
            )
          )
        )
      )
    )
  )
}

# Save to JSON/CSV.
dir.create("tests/data", recursive = TRUE, showWarnings = FALSE)
write_json(out1, "tests/data/intercept_only.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out2, "tests/data/random_slopes.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out3, "tests/data/penicillin.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_pastes_cask, "tests/data/pastes_cask_reml.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out4, "tests/data/mock_ml.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_binom, "tests/data/glmm_binomial.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_binom_probit, "tests/data/glmm_binomial_probit.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_binom_wt, "tests/data/glmm_binomial_weighted.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_pois, "tests/data/glmm_poisson.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_pois_off, "tests/data/glmm_poisson_offset.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_offset, "tests/data/sleepstudy_offset_reml.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(out_orange, "tests/data/orange_nlmer.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write_json(build_golden_manifest(), "tests/data/golden_parity_manifest.json", pretty = TRUE, auto_unbox = FALSE, digits = NA)
write.csv(sleepstudy, "tests/data/sleepstudy.csv", row.names = FALSE)
write.csv(Penicillin, "tests/data/penicillin.csv", row.names = FALSE)
write.csv(cbpp_binary, "tests/data/cbpp_binary.csv", row.names = FALSE)
write.csv(cbpp_binary[, c("herd", "period2", "period3", "period4", "y", "prior_w")], "tests/data/cbpp_binary_weighted.csv", row.names = FALSE)
write.csv(grouseticks, "tests/data/grouseticks.csv", row.names = FALSE)
cat("Successfully generated test data and golden parity manifest.\n")
