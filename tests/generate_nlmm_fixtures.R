#!/usr/bin/env Rscript
# Regenerate SSmicmen / SSgompertz golden fixtures from lme4 (uses user R library, not repo rlib).
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.6")
.libPaths(c(user_lib, .libPaths()))
library(lme4)
library(jsonlite)

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
cat("Wrote tests/data/ssmicmen_*.csv/json, ssgompertz_*.csv/json, sspower_*.csv/json\n")
cat("Update golden_parity_manifest.json expected values manually or via tests/generate_test_data.R\n")
