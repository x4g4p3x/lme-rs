#!/usr/bin/env Rscript
# Format or check R comparison / golden-parity scripts (invoked from lme_ci.py).
args <- commandArgs(trailingOnly = TRUE)
check <- "--check" %in% args
files <- setdiff(args, "--check")

if (length(files) == 0) {
  quit(status = 0)
}

suppressPackageStartupMessages(library(styler))

dry <- if (check) "fail" else "off"
for (path in files) {
  if (!file.exists(path)) {
    stop("missing file: ", path, call. = FALSE)
  }
  style_file(path, dry = dry)
}
