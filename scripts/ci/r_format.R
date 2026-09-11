#!/usr/bin/env Rscript
# Format or check R comparison / golden-parity scripts (invoked from lme_ci.py).
# An inherited Unix locale can leave Windows R in the C locale. styler then
# rewrites Unicode in comments and strings as literal <U+....> text.
if (.Platform$OS.type == "windows" && !l10n_info()[["UTF-8"]]) {
  invisible(Sys.setlocale("LC_CTYPE", ".UTF-8"))
  stopifnot(l10n_info()[["UTF-8"]])
}
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
