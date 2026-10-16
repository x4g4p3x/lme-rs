# Independent reference fits for project-owned rational boundary fixtures.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: lmm_boundary_checks.R FIXTURES OUTPUT")
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(pbkrtest))
lock <- jsonlite::read_json("scientific/renv.lock")
if (as.character(getRversion()) != lock$R$Version) stop("R version drift")
for (name in c("lme4", "lmerTest", "pbkrtest", "jsonlite")) {
  if (packageDescription(name, fields = "Version") != lock$Packages[[name]]$Version) stop("reference package drift")
}
if (file.exists(args[[2]])) stop("output exists")
cases <- jsonlite::read_json(args[[1]])$fixtures
results <- lapply(cases, function(case) {
  data <- data.frame(
    y = unlist(case$observations), x = rep(unlist(case$group_x), each = 2),
    group = factor(rep(1:4, each = 2))
  )
  fit <- suppressMessages(lmerTest::lmer(y ~ x + (1 | group), data,
    REML = TRUE,
    control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
  ))
  kr <- pbkrtest::KRmodcomp(fit, matrix(c(0, 1), nrow = 1))$stats
  sat <- lmerTest::contest1D(fit, c(0, 1), ddf = "Satterthwaite")
  list(
    case = case$id, converged = all(fit@optinfo$conv$opt == 0),
    beta0 = unname(fixef(fit)[[1]]), beta1 = unname(fixef(fit)[[2]]),
    sigma2 = sigma(fit)^2, tau2 = sigma(fit)^2 * unname(lme4::getME(fit, "theta")[[1]])^2,
    slope_variance = unname(vcov(fit)[2, 2]), criterion = -2 * as.numeric(logLik(fit)),
    kr_variance = unname(pbkrtest::vcovAdj(fit)[2, 2]), kr_df = kr$ddf, kr_p = kr$p.value,
    satterthwaite_df = sat$df, satterthwaite_p = sat[["Pr(>|t|)"]],
    group_means_p = pf(case$expected$beta1^2 / case$expected$group_means_variance,
      1, case$expected$q,
      lower.tail = FALSE
    )
  )
})
jsonlite::write_json(list(
  r_version = R.version.string,
  packages = setNames(lapply(c("lme4", "lmerTest", "pbkrtest"), function(x) as.character(packageVersion(x))), c("lme4", "lmerTest", "pbkrtest")),
  results = results
), args[[2]], auto_unbox = TRUE, pretty = TRUE, digits = 17)
