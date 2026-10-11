# Project-owned adapter: evaluate only the declared worked-example calls from
# the hash-verified, publicly downloaded author scripts. No implementation is
# translated into the Rust engine. Dataset exports stay in the run directory.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: paper_references.R RUN_DIRECTORY")
out <- normalizePath(args[[1]], mustWork = TRUE)
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(pbkrtest))

author_lines <- function(name, first, last) {
  text <- readLines(file.path(out, name), warn = FALSE)
  eval(parse(text = text[first:last]), envir = .GlobalEnv)
}
write_data <- function(data, name) {
  write.csv(data, file.path(out, name), row.names = FALSE, quote = TRUE)
}
fit_record <- function(fit) {
  list(
    coefficients = as.list(lme4::fixef(fit)),
    std_errors = as.list(sqrt(diag(vcov(fit)))),
    sigma2 = sigma(fit)^2,
    theta = unname(lme4::getME(fit, "theta")),
    theta_names = names(lme4::getME(fit, "theta")),
    criterion = -2 * as.numeric(logLik(fit)),
    fitted = unname(fitted(fit)),
    covariance = unname(as.matrix(vcov(fit))),
    optimizer_messages = fit@optinfo$conv$lme4$messages
  )
}
anova_record <- function(table) {
  lapply(seq_len(nrow(table)), function(i) {
    list(
      term = rownames(table)[i], num_df = table[i, "NumDF"],
      den_df = table[i, "DenDF"], f_value = table[i, "F value"],
      p_value = table[i, "Pr(>F)"]
    )
  })
}

# Bates et al. (2015), Section 1.2; run the exact author fit statement.
author_lines("bates-2015.R", 19L, 19L)
write_data(lme4::sleepstudy, "sleepstudy.csv")
bates <- fit_record(fm1)

# Kuznetsova et al. (2017), Section 8.2; the second fit uses lmerTest.
author_lines("kuznetsova-2017.R", 90L, 106L)
write_data(TVbo[c("Sharpnessofmovement", "TVset", "Picture", "Assessor")], "TVbo.csv")
kuznetsova <- list(
  fit = fit_record(tv),
  satterthwaite_type3 = anova_record(anova(tv)),
  kenward_roger_type2 = anova_record(anova(tv, type = 2, ddf = "Kenward-Roger"))
)

# Halekoh and Hojsgaard (2014), beets fit and the harvest KR comparison.
# KRmodcomp refits the author's ML models using REML internally.
author_lines("halekoh-2014.R", 26L, 30L)
author_lines("halekoh-2014.R", 79L, 79L)
write_data(beets[c("sugpct", "block", "sow", "harvest")], "beets.csv")
kr <- sug.kr.h$stats
halekoh <- list(
  fit = fit_record(update(sug, REML = TRUE)),
  harvest_kr = list(
    num_df = kr$ndf, den_df = kr$ddf, f_value = kr$Fstat,
    p_value = kr$p.value, scaling = kr$F.scaling
  )
)
records <- list(
  schema_version = 1L,
  scope = "selected deterministic worked examples; not whole-paper replication",
  r_version = R.version.string,
  packages = setNames(
    lapply(
      c("lme4", "lmerTest", "pbkrtest", "jsonlite"),
      function(x) as.character(packageVersion(x))
    ),
    c("lme4", "lmerTest", "pbkrtest", "jsonlite")
  ),
  namespace_versions = setNames(
    lapply(
      sort(loadedNamespaces()),
      function(x) as.character(packageVersion(x))
    ),
    sort(loadedNamespaces())
  ),
  contrasts = getOption("contrasts"),
  bates = bates, kuznetsova = kuznetsova, halekoh = halekoh
)
jsonlite::write_json(records, file.path(out, "paper-references.json"),
  pretty = TRUE, auto_unbox = TRUE, digits = 17, null = "null"
)
writeLines(capture.output(sessionInfo()), file.path(out, "R-session.txt"))
