# Shared setup for parity R exporters (local library path + jsonlite).
dir.create("rlib", showWarnings = FALSE)
.libPaths(c("rlib", .libPaths()))
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org", lib = "rlib")
}
