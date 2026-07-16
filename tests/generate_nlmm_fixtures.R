#!/usr/bin/env Rscript
# Regenerate SSmicmen / SSgompertz golden fixtures from lme4 (uses user R library, not repo rlib).
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.6")
.libPaths(c(user_lib, .libPaths()))
library(lme4)
library(jsonlite)

safe_nlmer <- function(expr, label) {
  tryCatch(expr, error = function(e) {
    cat(sprintf("SKIP %s: %s\n", label, conditionMessage(e)))
    NULL
  })
}

# SSmicmen
set.seed(2025)
m_mic <- 5
n_mic <- 12
id_mic <- factor(rep(1:m_mic, each = n_mic))
x_mic <- rep(seq(0.5, 6, length.out = n_mic), m_mic)
b_mic <- rnorm(m_mic, 0, 0.5)
y_mic <- 12 * x_mic / (2 + x_mic) + b_mic[as.integer(id_mic)] + rnorm(length(x_mic), 0, 0.3)
df_micmen <- data.frame(y = y_mic, x = x_mic, id = id_mic)
write.csv(df_micmen, "tests/data/ssmicmen_synthetic.csv", row.names = FALSE)
start_mic <- getInitial(y ~ SSmicmen(x, Vmax, K), data = df_micmen)
fm_micmen <- nlmer(y ~ SSmicmen(x, Vmax, K) ~ Vmax | id, data = df_micmen, start = start_mic)
micmen_vc_sd <- as.numeric(attr(VarCorr(fm_micmen)$id, "stddev")["Vmax"])
out_micmen <- list(
  model = "y ~ SSmicmen(x, Vmax, K) ~ Vmax|id",
  outputs = list(
    beta = as.numeric(fixef(fm_micmen)),
    theta = as.numeric(getME(fm_micmen, "theta")),
    re_sd = micmen_vc_sd,
    sigma2 = as.numeric(sigma(fm_micmen))^2,
    logLik = as.numeric(logLik(fm_micmen))
  )
)
write_json(out_micmen, "tests/data/ssmicmen_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("SSmicmen fixef:", paste(signif(fixef(fm_micmen), 6), collapse = ", "), "\n")
cat("SSmicmen sd:", micmen_vc_sd, "\n")

# SSgompertz (stats::SSgompertz: Asym * exp(-b2 * b3^x))
set.seed(2026)
m_gom <- 5
n_gom <- 12
id_gom <- factor(rep(1:m_gom, each = n_gom))
x_gom <- rep(seq(0, 4, length.out = n_gom), m_gom)
Asym_g <- 50
b2_g <- 2.0
b3_g <- 0.3
b_gom <- rnorm(m_gom, 0, 2)
y_gom <- Asym_g * exp(-b2_g * b3_g^x_gom) + b_gom[as.integer(id_gom)] + rnorm(length(x_gom), 0, 0.8)
df_gompertz <- data.frame(y = y_gom, x = x_gom, id = id_gom)
write.csv(df_gompertz, "tests/data/ssgompertz_synthetic.csv", row.names = FALSE)
start_gom <- getInitial(y ~ SSgompertz(x, Asym, b2, b3), data = df_gompertz)
fm_gompertz <- nlmer(y ~ SSgompertz(x, Asym, b2, b3) ~ Asym | id, data = df_gompertz, start = start_gom)
gompertz_vc_sd <- as.numeric(attr(VarCorr(fm_gompertz)$id, "stddev")["Asym"])
out_gompertz <- list(
  model = "y ~ SSgompertz(x, Asym, b2, b3) ~ Asym|id",
  outputs = list(
    beta = as.numeric(fixef(fm_gompertz)),
    theta = as.numeric(getME(fm_gompertz, "theta")),
    re_sd = gompertz_vc_sd,
    sigma2 = as.numeric(sigma(fm_gompertz))^2,
    logLik = as.numeric(logLik(fm_gompertz))
  )
)
write_json(out_gompertz, "tests/data/ssgompertz_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("SSgompertz fixef:", paste(signif(fixef(fm_gompertz), 6), collapse = ", "), "\n")
cat("SSgompertz sd:", gompertz_vc_sd, "\n")
# SSpower (MATLAB power2: a * x^b + c; lme-rs extension — custom selfStart in R for lme4 parity)
initPower2 <- function(mCall, data, LHS, ...) {
  xy <- sortedXyData(mCall[["x"]], LHS, data)
  xv <- xy[, "x"]
  yv <- xy[, "y"]
  ok <- xv > 0 & is.finite(xv) & is.finite(yv)
  xv <- xv[ok]
  yv <- yv[ok]
  if (length(xv) < 3) {
    return(c(a = 1, b = 1, c = 0))
  }
  c_est <- min(yv) - 0.05 * max(abs(min(yv)), 1)
  ok2 <- (yv - c_est) > 0
  if (sum(ok2) < 2) {
    return(c(a = 1, b = 1, c = c_est))
  }
  fit <- lm(I(log(yv[ok2] - c_est)) ~ I(log(xv[ok2])))
  c(
    a = max(exp(unname(coef(fit)[1])), 1e-8),
    b = unname(coef(fit)[2]),
    c = c_est
  )
}
SSpower <- function(x, a, b, c) {
  xb <- x^b
  val <- a * xb + c
  grad <- cbind(a = xb, b = a * xb * log(x), c = 1)
  attr(val, "gradient") <- grad
  val
}
attr(SSpower, "initial") <- initPower2
attr(SSpower, "pnames") <- c("a", "b", "c")
class(SSpower) <- "selfStart"

set.seed(2027)
m_pwr <- 5
n_pwr <- 12
id_pwr <- factor(rep(1:m_pwr, each = n_pwr))
x_pwr <- rep(seq(0.5, 5, length.out = n_pwr), m_pwr)
b_pwr <- rnorm(m_pwr, 0, 0.15)
y_pwr <- 2 * x_pwr^0.5 + 1 + b_pwr[as.integer(id_pwr)] + rnorm(length(x_pwr), 0, 0.15)
df_power <- data.frame(y = y_pwr, x = x_pwr, id = id_pwr)
write.csv(df_power, "tests/data/sspower_synthetic.csv", row.names = FALSE)
start_pwr <- getInitial(y ~ SSpower(x, a, b, c), data = df_power)
fm_power <- nlmer(y ~ SSpower(x, a, b, c) ~ c | id, data = df_power, start = start_pwr)
power_vc_sd <- as.numeric(attr(VarCorr(fm_power)$id, "stddev")["c"])
out_power <- list(
  model = "y ~ SSpower(x, a, b, c) ~ c|id",
  outputs = list(
    beta = as.numeric(fixef(fm_power)),
    theta = as.numeric(getME(fm_power, "theta")),
    re_sd = power_vc_sd,
    sigma2 = as.numeric(sigma(fm_power))^2,
    logLik = as.numeric(logLik(fm_power))
  )
)
write_json(out_power, "tests/data/sspower_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("SSpower fixef:", paste(signif(fixef(fm_power), 6), collapse = ", "), "\n")
cat("SSpower sd:", power_vc_sd, "\n")

# SSfpl
set.seed(2030)
m_fpl <- 5
n_fpl <- 12
id_fpl <- factor(rep(1:m_fpl, each = n_fpl))
x_fpl <- rep(seq(0, 10, length.out = n_fpl), m_fpl)
b_fpl <- rnorm(m_fpl, 0, 1.5)
y_fpl <- SSfpl(x_fpl, 10, 50, 5, 2) + b_fpl[as.integer(id_fpl)] + rnorm(length(x_fpl), 0, 0.8)
df_fpl <- data.frame(y = y_fpl, x = x_fpl, id = id_fpl)
write.csv(df_fpl, "tests/data/ssfpl_synthetic.csv", row.names = FALSE)
start_fpl <- getInitial(y ~ SSfpl(x, A, B, xmid, scal), data = df_fpl)
fm_fpl <- nlmer(y ~ SSfpl(x, A, B, xmid, scal) ~ A | id, data = df_fpl, start = start_fpl)
fpl_vc_sd <- as.numeric(attr(VarCorr(fm_fpl)$id, "stddev")["A"])
out_fpl <- list(
  model = "y ~ SSfpl(x, A, B, xmid, scal) ~ A|id",
  outputs = list(
    beta = as.numeric(fixef(fm_fpl)),
    theta = as.numeric(getME(fm_fpl, "theta")),
    re_sd = fpl_vc_sd,
    sigma2 = as.numeric(sigma(fm_fpl))^2,
    logLik = as.numeric(logLik(fm_fpl))
  )
)
write_json(out_fpl, "tests/data/ssfpl_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("SSfpl fixef:", paste(signif(fixef(fm_fpl), 6), collapse = ", "), "\n")

# SSbiexp — quiet DGP + truth start (noisy / poorly separated rates trip PIRLS)
set.seed(42)
m_bi <- 6
n_bi <- 16
id_bi <- factor(rep(seq_len(m_bi), each = n_bi))
x_bi <- rep(seq(0.1, 5, length.out = n_bi), m_bi)
b_bi <- rnorm(m_bi, 0, 0.08)
y_bi <- SSbiexp(x_bi, 5, log(1.2), 3, log(0.3)) + b_bi[as.integer(id_bi)] + rnorm(length(x_bi), 0, 0.02)
df_bi <- data.frame(y = y_bi, x = x_bi, id = id_bi)
write.csv(df_bi, "tests/data/ssbiexp_synthetic.csv", row.names = FALSE)
start_bi <- c(A1 = 5, lrc1 = log(1.2), A2 = 3, lrc2 = log(0.3))
fm_bi <- safe_nlmer(
  nlmer(y ~ SSbiexp(x, A1, lrc1, A2, lrc2) ~ A1 | id, data = df_bi, start = start_bi),
  "SSbiexp nlmer"
)
if (!is.null(fm_bi)) {
  bi_vc_sd <- as.numeric(attr(VarCorr(fm_bi)$id, "stddev")["A1"])
  out_bi <- list(
    model = "y ~ SSbiexp(x, A1, lrc1, A2, lrc2) ~ A1|id",
    outputs = list(
      beta = as.numeric(fixef(fm_bi)),
      theta = as.numeric(getME(fm_bi, "theta")),
      re_sd = bi_vc_sd,
      sigma2 = as.numeric(sigma(fm_bi))^2,
      logLik = as.numeric(logLik(fm_bi))
    )
  )
  write_json(out_bi, "tests/data/ssbiexp_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("SSbiexp fixef:", paste(signif(fixef(fm_bi), 6), collapse = ", "), "\n")
} else {
  cat("SSbiexp: wrote CSV only (no nlmer golden)\n")
}

# SSweibull — quiet DGP, x>0, truth start
set.seed(99)
m_w <- 6
n_w <- 18
id_w <- factor(rep(seq_len(m_w), each = n_w))
x_w <- rep(seq(0.2, 5, length.out = n_w), m_w)
b_w <- rnorm(m_w, 0, 0.8)
y_w <- SSweibull(x_w, 100, 80, -1, 1.5) + b_w[as.integer(id_w)] + rnorm(length(x_w), 0, 0.3)
df_w <- data.frame(y = y_w, x = x_w, id = id_w)
write.csv(df_w, "tests/data/ssweibull_synthetic.csv", row.names = FALSE)
start_w <- c(Asym = 100, Drop = 80, lrc = -1, pwr = 1.5)
fm_w <- safe_nlmer(
  nlmer(y ~ SSweibull(x, Asym, Drop, lrc, pwr) ~ Asym | id, data = df_w, start = start_w),
  "SSweibull nlmer"
)
if (!is.null(fm_w)) {
  w_vc_sd <- as.numeric(attr(VarCorr(fm_w)$id, "stddev")["Asym"])
  out_w <- list(
    model = "y ~ SSweibull(x, Asym, Drop, lrc, pwr) ~ Asym|id",
    outputs = list(
      beta = as.numeric(fixef(fm_w)),
      theta = as.numeric(getME(fm_w, "theta")),
      re_sd = w_vc_sd,
      sigma2 = as.numeric(sigma(fm_w))^2,
      logLik = as.numeric(logLik(fm_w))
    )
  )
  write_json(out_w, "tests/data/ssweibull_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("SSweibull fixef:", paste(signif(fixef(fm_w), 6), collapse = ", "), "\n")
} else {
  cat("SSweibull: wrote CSV only (no nlmer golden)\n")
}

# SSasympOff
set.seed(2033)
m_ao <- 5
n_ao <- 12
id_ao <- factor(rep(1:m_ao, each = n_ao))
x_ao <- rep(seq(0, 5, length.out = n_ao), m_ao)
b_ao <- rnorm(m_ao, 0, 2)
y_ao <- SSasympOff(x_ao, 90, log(0.4), 0.5) + b_ao[as.integer(id_ao)] + rnorm(length(x_ao), 0, 1.0)
df_ao <- data.frame(y = y_ao, x = x_ao, id = id_ao)
write.csv(df_ao, "tests/data/ssasympoff_synthetic.csv", row.names = FALSE)
start_ao <- tryCatch(
  getInitial(y ~ SSasympOff(x, Asym, lrc, c0), data = df_ao),
  error = function(e) c(Asym = 90, lrc = log(0.4), c0 = 0.5)
)
fm_ao <- safe_nlmer(
  nlmer(y ~ SSasympOff(x, Asym, lrc, c0) ~ Asym | id, data = df_ao, start = start_ao),
  "SSasympOff nlmer"
)
if (!is.null(fm_ao)) {
  ao_vc_sd <- as.numeric(attr(VarCorr(fm_ao)$id, "stddev")["Asym"])
  out_ao <- list(
    model = "y ~ SSasympOff(x, Asym, lrc, c0) ~ Asym|id",
    outputs = list(
      beta = as.numeric(fixef(fm_ao)),
      theta = as.numeric(getME(fm_ao, "theta")),
      re_sd = ao_vc_sd,
      sigma2 = as.numeric(sigma(fm_ao))^2,
      logLik = as.numeric(logLik(fm_ao))
    )
  )
  write_json(out_ao, "tests/data/ssasympoff_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("SSasympOff fixef:", paste(signif(fixef(fm_ao), 6), collapse = ", "), "\n")
}

# SSasympOrig
set.seed(2034)
m_ar <- 5
n_ar <- 12
id_ar <- factor(rep(1:m_ar, each = n_ar))
x_ar <- rep(seq(0, 5, length.out = n_ar), m_ar)
b_ar <- rnorm(m_ar, 0, 2)
y_ar <- SSasympOrig(x_ar, 90, log(0.4)) + b_ar[as.integer(id_ar)] + rnorm(length(x_ar), 0, 1.0)
df_ar <- data.frame(y = y_ar, x = x_ar, id = id_ar)
write.csv(df_ar, "tests/data/ssasymporig_synthetic.csv", row.names = FALSE)
start_ar <- tryCatch(
  getInitial(y ~ SSasympOrig(x, Asym, lrc), data = df_ar),
  error = function(e) c(Asym = 90, lrc = log(0.4))
)
fm_ar <- safe_nlmer(
  nlmer(y ~ SSasympOrig(x, Asym, lrc) ~ Asym | id, data = df_ar, start = start_ar),
  "SSasympOrig nlmer"
)
if (!is.null(fm_ar)) {
  ar_vc_sd <- as.numeric(attr(VarCorr(fm_ar)$id, "stddev")["Asym"])
  out_ar <- list(
    model = "y ~ SSasympOrig(x, Asym, lrc) ~ Asym|id",
    outputs = list(
      beta = as.numeric(fixef(fm_ar)),
      theta = as.numeric(getME(fm_ar, "theta")),
      re_sd = ar_vc_sd,
      sigma2 = as.numeric(sigma(fm_ar))^2,
      logLik = as.numeric(logLik(fm_ar))
    )
  )
  write_json(out_ar, "tests/data/ssasymporig_nlmer.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("SSasympOrig fixef:", paste(signif(fixef(fm_ar), 6), collapse = ", "), "\n")
}

# CBPP AGQ (nAGQ = 7) — expanded binary trials
if (file.exists("tests/data/cbpp_binary.csv")) {
  cbpp_binary <- read.csv("tests/data/cbpp_binary.csv")
  fm_agq <- glmer(
    y ~ period2 + period3 + period4 + (1 | herd),
    data = cbpp_binary,
    family = binomial,
    nAGQ = 7
  )
  out_agq <- list(
    model = "y ~ period2 + period3 + period4 + (1 | herd) [Binomial nAGQ=7]",
    outputs = list(
      beta = as.numeric(fixef(fm_agq)),
      theta = as.numeric(getME(fm_agq, "theta")),
      deviance = as.numeric(deviance(fm_agq)),
      logLik = as.numeric(logLik(fm_agq))
    )
  )
  write_json(out_agq, "tests/data/glmm_binomial_agq.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("CBPP AGQ fixef:", paste(signif(fixef(fm_agq), 6), collapse = ", "), "\n")
  cat("CBPP AGQ theta:", paste(signif(getME(fm_agq, "theta"), 6), collapse = ", "), "\n")
}

# Sleepstudy profile confint (ML)
if (file.exists("tests/data/sleepstudy.csv")) {
  sleep <- read.csv("tests/data/sleepstudy.csv")
  fm_ss <- lmer(Reaction ~ Days + (1 | Subject), data = sleep, REML = FALSE)
  ci <- tryCatch(
    confint(fm_ss, parm = "beta_", method = "profile", level = 0.95, oldNames = FALSE),
    error = function(e) confint(fm_ss, method = "profile", level = 0.95)
  )
  rn <- rownames(ci)
  keep <- grepl("Intercept|Days", rn, ignore.case = TRUE)
  if (!any(keep)) keep <- seq_len(min(2L, nrow(ci)))
  ci_fe <- ci[keep, , drop = FALSE]
  out_ci <- list(
    model = "Reaction ~ Days + (1 | Subject) ML",
    level = 0.95,
    method = "profile",
    outputs = list(
      names = rownames(ci_fe),
      lower = as.numeric(ci_fe[, 1]),
      upper = as.numeric(ci_fe[, 2]),
      estimate = as.numeric(fixef(fm_ss))
    )
  )
  write_json(out_ci, "tests/data/sleepstudy_confint_profile.json", pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("Sleepstudy profile CI:\n")
  print(ci_fe)
}

cat("Wrote additional SS*/AGQ/profile fixtures.\n")
cat("Update golden_parity_manifest.json expected values manually or via tests/generate_test_data.R\n")
