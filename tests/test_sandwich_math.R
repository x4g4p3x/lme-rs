# Create a local library directory to circumvent permission errors
dir.create("rlib", showWarnings = FALSE)
.libPaths(c("rlib", .libPaths()))

if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4", repos = "https://cloud.r-project.org", lib = "rlib")
}
if (!requireNamespace("clubSandwich", quietly = TRUE)) {
  install.packages("clubSandwich", repos = "https://cloud.r-project.org", lib = "rlib")
}

library(lme4)
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)

# Manual calculation using V^{-1}(Y - X\beta) ?= \hat{\epsilon}
# Actually, let's see if the Meat M = \sum_j (X_j^T eps_j) (X_j^T eps_j)^T
# matches the clubSandwich CR0 meat.

X <- getME(fm1, "X")
eps <- resid(fm1) # conditional residuals: Y - X\beta - Zb
inv_info <- summary(fm1)$vcov / sigma(fm1)^2 # this is (X^T V^{-1} X)^{-1}

subjects <- sleepstudy$Subject
M <- matrix(0, nrow=2, ncol=2)
for (sub in unique(subjects)) {
  idx <- which(subjects == sub)
  X_j <- X[idx, , drop=FALSE]
  eps_j <- eps[idx]
  g_j <- t(X_j) %*% eps_j
  M <- M + g_j %*% t(g_j)
}

V_robust_manual <- inv_info %*% M %*% inv_info

cat("\nManual CR0 using conditional residuals:\n")
print(V_robust_manual)

# Compare to clubSandwich
if (requireNamespace("clubSandwich", quietly = TRUE)) {
  library(clubSandwich)
  vcov_cr0 <- vcovCR(fm1, type="CR0")
  cat("\nclubSandwich CR0:\n")
  print(vcov_cr0)
} else {
  cat("\nclubSandwich package not installed.\n")
}
