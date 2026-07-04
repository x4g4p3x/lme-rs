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
cat("Wrote tests/data/ssmicmen_*.csv/json and ssgompertz_*.csv/json\n")
cat("Update golden_parity_manifest.json expected values manually or via tests/generate_test_data.R\n")
