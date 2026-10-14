# Verify or restore the locked R 4.6.1 scientific reference environment.
args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) >= 1L) args[[1L]] else "--verify"
lock_path <- "scientific/renv.lock"
target_lib <- NULL

if (identical(mode, "--verify")) {
  if (length(args) >= 2L) lock_path <- args[[2L]]
} else if (identical(mode, "--restore")) {
  if (length(args) < 2L) {
    stop("usage: restore_r_environment.R --restore TARGET_LIBRARY [LOCKFILE]")
  }
  target_lib <- args[[2L]]
  dir.create(target_lib, recursive = TRUE, showWarnings = FALSE)
  target_lib <- normalizePath(target_lib, mustWork = TRUE)
  .libPaths(c(target_lib, .libPaths()))
  if (length(args) >= 3L) lock_path <- args[[3L]]
} else {
  stop("usage: restore_r_environment.R [--verify [LOCKFILE] | --restore TARGET_LIBRARY [LOCKFILE]]")
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  if (is.null(target_lib)) stop("jsonlite is required to read scientific/renv.lock")
  install.packages(
    "https://cloud.r-project.org/src/contrib/jsonlite_2.0.0.tar.gz",
    lib = target_lib, repos = NULL, type = "source", quiet = TRUE
  )
}

lock <- jsonlite::read_json(lock_path, simplifyVector = TRUE)
if (as.character(getRversion()) != lock$R$Version) {
  stop(paste("R version drift: expected", lock$R$Version, "got", getRversion()))
}

installed_exact <- function(pkg, raw_version, lib_loc = NULL) {
  desc <- suppressWarnings(packageDescription(pkg, lib.loc = lib_loc, fields = "Version"))
  !is.na(desc) && identical(as.character(desc), as.character(raw_version))
}

install_exact <- function(pkg, raw_version, repo_url, lib_dir) {
  archive <- sprintf("%s_%s.tar.gz", pkg, raw_version)
  urls <- c(
    sprintf("%s/src/contrib/%s", repo_url, archive),
    sprintf("%s/src/contrib/Archive/%s/%s", repo_url, pkg, archive)
  )
  dest <- file.path(tempdir(), archive)
  on.exit(unlink(dest), add = TRUE)
  downloaded <- FALSE
  for (url in urls) {
    status <- tryCatch(
      utils::download.file(url, dest, mode = "wb", quiet = TRUE),
      error = function(e) 1L
    )
    if (identical(status, 0L) && file.exists(dest) && file.info(dest)$size > 0) {
      downloaded <- TRUE
      break
    }
  }
  if (!downloaded) {
    stop(paste("failed to download locked CRAN source archive for", pkg, raw_version))
  }
  utils::install.packages(dest, lib = lib_dir, repos = NULL, type = "source", quiet = TRUE)
  if (!installed_exact(pkg, raw_version, lib_loc = lib_dir)) {
    stop(paste("restored package version mismatch for", pkg))
  }
}

if (identical(mode, "--restore")) {
  repo_url <- lock$R$Repositories$URL[[1L]]
  for (pkg in lock$InstallOrder) {
    entry <- lock$Packages[[pkg]]
    if (!installed_exact(pkg, entry$Version)) {
      cat(sprintf("restoring %s %s\n", pkg, entry$Version))
      install_exact(pkg, entry$Version, repo_url, target_lib)
    }
  }
}

for (pkg in names(lock$Packages)) {
  entry <- lock$Packages[[pkg]]
  raw <- suppressWarnings(packageDescription(pkg, fields = "Version"))
  if (is.na(raw) || !identical(as.character(raw), as.character(entry$Version))) {
    stop(paste("package raw version drift:", pkg, "expected", entry$Version, "got", raw))
  }
  norm <- as.character(packageVersion(pkg))
  if (!identical(norm, as.character(entry$NormalizedVersion))) {
    stop(paste("package version drift:", pkg, "expected", entry$NormalizedVersion, "got", norm))
  }
}

suppressPackageStartupMessages({
  library(lmerTest)
  library(pbkrtest)
  library(jsonlite)
})

for (ns in names(lock$LoadedNamespaces)) {
  expected <- as.character(lock$LoadedNamespaces[[ns]])
  actual <- as.character(packageVersion(ns))
  if (!identical(actual, expected)) {
    stop(paste("loaded namespace version drift:", ns, "expected", expected, "got", actual))
  }
}

cat(sprintf(
  "scientific R environment verified: R %s, %d locked packages, %d namespaces\n",
  as.character(getRversion()), length(lock$Packages), length(lock$LoadedNamespaces)
))
