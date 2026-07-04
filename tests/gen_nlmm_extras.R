dir.create("rlib", showWarnings = FALSE)
.libPaths(c("rlib", .libPaths()))
if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4", lib = "rlib", repos = "https://cloud.r-project.org")
}
library(lme4)

set.seed(2024)
m <- 5
n_per <- 12
id <- factor(rep(1:m, each = n_per))
x <- rep(seq(0, 3, length.out = n_per), m)
Asym <- 100
R0 <- 15
lrc <- log(0.5)
b_id <- rnorm(m, 0, 5)
y <- Asym - (Asym - R0) * exp(-exp(lrc) * x) + b_id[as.integer(id)] + rnorm(length(x), 0, 1.5)
df <- data.frame(y = y, x = x, id = id)
write.csv(df, "tests/data/ssasymp_synthetic.csv", row.names = FALSE)

# SSmicmen / SSgompertz synthetic CSVs (also written by generate_test_data.R when regenerated).
set.seed(2025)
m <- 5
n_per <- 12
id <- factor(rep(1:m, each = n_per))
x_mic <- rep(seq(0.5, 6, length.out = n_per), m)
b_mic <- rnorm(m, 0, 0.5)
y_mic <- 12 * x_mic / (2 + x_mic) + b_mic[as.integer(id)] + rnorm(length(x_mic), 0, 0.3)
write.csv(data.frame(y = y_mic, x = x_mic, id = id), "tests/data/ssmicmen_synthetic.csv", row.names = FALSE)

set.seed(2026)
x_gom <- rep(seq(0, 4, length.out = n_per), m)
b_gom <- rnorm(m, 0, 2)
y_gom <- 50 * exp(-2.0 * 0.3^x_gom) + b_gom[as.integer(id)] + rnorm(length(x_gom), 0, 0.8)
write.csv(data.frame(y = y_gom, x = x_gom, id = id), "tests/data/ssgompertz_synthetic.csv", row.names = FALSE)

fm <- nlmer(
  y ~ SSasymp(x, Asym, R0, lrc) ~ Asym | id,
  data = df,
  start = c(Asym = 90, R0 = 20, lrc = log(0.4))
)
cat("SSasymp fixef:", paste(signif(fixef(fm), 6), collapse = ", "), "\n")
cat("SSasymp sd:", attr(VarCorr(fm)[["id"]], "stddev"), "\n")

data(Orange)
Orange$Tree <- factor(Orange$Tree)
fm2 <- nlmer(
  circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree,
  data = Orange,
  start = c(Asym = 200, xmid = 725, scal = 350)
)
cat("Orange multi fixef:", paste(signif(fixef(fm2), 5), collapse = ", "), "\n")
cat("Orange multi sd:", paste(signif(attr(VarCorr(fm2)[["Tree"]], "stddev"), 5), collapse = ", "), "\n")
cat("Orange multi sigma:", sigma(fm2), "\n")
