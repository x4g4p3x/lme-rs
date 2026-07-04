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
