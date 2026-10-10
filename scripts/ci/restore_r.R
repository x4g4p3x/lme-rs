args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Expected the R environment project directory")
if (getRversion() != "4.6.1") stop("The locked comparison environment requires R 4.6.1")
project <- normalizePath(args[[1L]], mustWork = TRUE)
Sys.setenv(RENV_PROJECT = project)
source(file.path(project, "renv", "activate.R"))
renv::restore(project = project, prompt = FALSE)
status <- renv::status(project = project)
if (!isTRUE(status$synchronized)) stop("R environment does not match renv.lock")
for (pkg in c(
  "lme4", "lmerTest", "car", "rlang", "styler", "jsonlite", "emmeans",
  "multcomp", "clubSandwich", "pbkrtest"
)) {
  stopifnot(requireNamespace(pkg, quietly = TRUE))
  cat(pkg, as.character(packageVersion(pkg)), "\n")
}
