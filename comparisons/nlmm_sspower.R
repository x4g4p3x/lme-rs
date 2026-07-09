#!/usr/bin/env Rscript
# lme4 reference fit for SSpower (MATLAB power2: a * x^b + c) grouped calibration.
# R has no stats::SSpower; this script defines a custom selfStart mean for nlmer parity.
# Julia MixedModels.jl has no nlmer equivalent — see comparisons/COMPARISONS.md.

user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.6")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))
if (dir.exists("rlib")) .libPaths(c("rlib", .libPaths()))

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))

initPower2 <- function(mCall, data, LHS, ...) {
  xy <- sortedXyData(mCall[["x"]], LHS, data)
  xv <- xy[, "x"]
  yv <- xy[, "y"]
  ok <- xv > 0 & is.finite(xv) & is.finite(yv)
  xv <- xv[ok]
  yv <- yv[ok]
  if (length(xv) < 3) return(c(a = 1, b = 1, c = 0))
  c_est <- min(yv) - 0.05 * max(abs(min(yv)), 1)
  ok2 <- (yv - c_est) > 0
  if (sum(ok2) < 2) return(c(a = 1, b = 1, c = c_est))
  fit <- lm(I(log(yv[ok2] - c_est)) ~ I(log(xv[ok2])))
  c(a = max(exp(unname(coef(fit)[1])), 1e-8), b = unname(coef(fit)[2]), c = c_est)
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

df <- read.csv("tests/data/sspower_synthetic.csv")
start <- getInitial(y ~ SSpower(x, a, b, c), data = df)
fm <- nlmer(y ~ SSpower(x, a, b, c) ~ c | id, data = df, start = start)

cat("lme4::nlmer SSpower (custom selfStart)\n")
cat("fixef:\n")
print(fixef(fm))
cat("VarCorr:\n")
print(VarCorr(fm))
cat("sigma:", sigma(fm), "\n")
