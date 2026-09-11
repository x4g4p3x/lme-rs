function formatSeconds(seconds) {
  if (seconds == null || Number.isNaN(seconds)) {
    return "n/a";
  }
  if (seconds >= 10) {
    return `${seconds.toFixed(2)} s`;
  }
  if (seconds >= 1) {
    return `${seconds.toFixed(3)} s`;
  }
  if (seconds >= 0.001) {
    return `${(seconds * 1000).toFixed(2)} ms`;
  }
  return `${(seconds * 1e6).toFixed(1)} µs`;
}

function formatRatio(value) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return `${value.toFixed(2)}×`;
}

function formatVsJulia(ratio) {
  if (ratio == null || Number.isNaN(ratio)) {
    return "n/a";
  }
  const label = formatRatio(ratio);
  if (ratio < 1) {
    return `${label} Julia (faster)`;
  }
  if (ratio > 1) {
    return `${label} Julia`;
  }
  return `${label} Julia (tied)`;
}

function cssImpl(name) {
  return String(name).replace(/[^a-z0-9_-]/gi, "_");
}

function caseLabel(name) {
  return ({
    sleepstudy_reml: "Sleepstudy · random slopes",
    sleepstudy_weighted_reml: "Sleepstudy · weighted",
    penicillin_crossed_reml: "Penicillin · crossed groups",
    pastes_nested_reml: "Pastes · nested groups",
    random_intercept_10k: "Random intercepts · 10k rows",
    random_intercept_50k: "Random intercepts · 50k rows",
    random_intercept_100k: "Random intercepts · 100k rows",
    large_random_slopes_100k: "Random slopes · 100k rows",
    crossed_20k: "Crossed groups · 20k rows",
    nested_10k: "Nested groups · 10k rows",
    cbpp_binomial_ml: "CBPP · binomial GLMM",
    grouseticks_poisson_ml: "Grouseticks · Poisson GLMM",
    sleepstudy_lmer: "Sleepstudy · complete LMM fit",
    sleepstudy_satterthwaite: "Satterthwaite inference",
    sleepstudy_kenward_roger: "Kenward–Roger inference",
    orange_nlmer: "Orange · nonlinear mixed model",
    sleepstudy: "Sleepstudy",
    pastes: "Pastes",
    cbpp: "CBPP · binomial GLMM",
    grouseticks: "Grouseticks · Poisson GLMM",
    categorical: "Categorical predictors",
    weighted: "Weighted LMM",
    gaussian_glmm: "Gaussian GLMM",
  })[name] || name.replaceAll("_", " ");
}

function addLink(links, href, label) {
  if (!href) {
    return;
  }
  links.push({ href, label });
}

function renderHeroLinks(data) {
  const heroLinks = document.getElementById("hero-links");
  const links = [];
  addLink(links, data.links?.report_url, "Full measurement report");
  const siteLinks = data.links || {};

  addLink(links, siteLinks.coverage_url, "Coverage map");
  addLink(links, siteLinks.methodology_url, "Methodology");
  addLink(links, siteLinks.optimizer_url, "Optimizer guide");
  addLink(links, siteLinks.fair_source_url, "Fit data");
  addLink(links, siteLinks.optimizer_source_url, "Optimizer data");
  addLink(links, siteLinks.release_url, "GitHub release");
  addLink(links, siteLinks.run_url, "Workflow run");
  if (siteLinks.asset_urls) {
    addLink(links, siteLinks.asset_urls.fair, "Fair JSON");
    addLink(links, siteLinks.asset_urls.cross_language, "Example-script JSON");
    addLink(links, siteLinks.asset_urls.criterion, "Criterion archive");
  }
  addLink(links, siteLinks.repo_url, "Source");

  heroLinks.replaceChildren(
    ...links.map((link) => {
      const anchor = document.createElement("a");
      anchor.href = link.href;
      anchor.textContent = link.label;
      anchor.target = "_blank";
      anchor.rel = "noreferrer";
      return anchor;
    }),
  );
}

function renderSummary(data) {
  const runMeta = document.getElementById("run-meta");
  const fair = data.fair || {};
  const generatedAt = fair.generated_at
    ? new Date(fair.generated_at).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "unknown date";
  const sha = fair.git_sha ? fair.git_sha.slice(0, 12) : "n/a";
  const headline = data.headline || {};
  const config = fair.config || {};
  runMeta.textContent = `${fair.label || "Fair harness"} · ${generatedAt} · ${sha} · ${config.warmups ?? "?"} warmups / ${config.repeats ?? "?"} repeats`;
  const stats = [
    ["Qualified comparisons", `${headline.cold_fit_cases} of ${(fair.cases || []).length}`],
    ["Cases favoring Rust", `${headline.cold_fit_passes} of ${headline.cold_fit_cases} qualified`],
    ["Geomean Rust/Julia", formatRatio(headline.geometric_mean_rust_over_julia_cold_fit)],
    ["Median Rust/Julia", formatRatio(headline.median_rust_over_julia_cold_fit)],
  ];

  const statsGrid = document.getElementById("stats-grid");
  statsGrid.replaceChildren(
    ...stats.map(([label, value]) => {
      const card = document.createElement("div");
      card.className = "stat-card";
      const labelNode = document.createElement("span");
      labelNode.className = "stat-label";
      labelNode.textContent = label;
      const valueNode = document.createElement("div");
      valueNode.className = "stat-value";
      valueNode.textContent = String(value);
      card.append(labelNode, valueNode);
      return card;
    }),
  );
}

function renderRatioSummary(data) {
  const container = document.getElementById("ratio-summary");
  const fair = data.fair || {};
  const summary = fair.summary || {};
  const rows = [
    ["Geometric mean · qualified cases", summary.geometric_mean_rust_over_julia_cold_fit],
    ["Median · qualified cases", summary.median_rust_over_julia_cold_fit],
  ];
  const maxRatio = Math.max(
    1,
    ...rows.map(([, value]) => value || 0).filter((value) => value > 0),
  );

  container.replaceChildren(
    ...rows.map(([label, ratio]) => {
      const row = document.createElement("div");
      row.className = "ratio-row";
      const name = document.createElement("div");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "ratio-track";
      const fill = document.createElement("div");
      fill.className = "ratio-fill";
      if (ratio != null && ratio < 1) {
        fill.classList.add("is-faster");
      }
      const width = ratio ? Math.max((Math.min(ratio, maxRatio) / maxRatio) * 100, 3) : 3;
      fill.style.width = `${width}%`;
      track.append(fill);
      const value = document.createElement("div");
      value.textContent = `${formatRatio(ratio)} Julia (descriptive aggregate)`;
      row.append(name, track, value);
      return row;
    }),
  );
}

function fillBars(barsNode, entries, valueText) {
  const rowTemplate = document.getElementById("bar-row-template");
  entries.forEach((entry) => {
    const row = rowTemplate.content.cloneNode(true);
    row.querySelector(".bar-label").textContent = entry.label || entry.implementation;
    const fill = row.querySelector(".bar-fill");
    fill.classList.add(cssImpl(entry.implementation));
    fill.style.width = `${Math.max((entry.width_fraction || 0) * 100, 2)}%`;
    row.querySelector(".bar-value").textContent = valueText(entry);
    barsNode.append(row);
  });
}

function renderFairCaseCards(target, cases, metricKey) {
  const caseTemplate = document.getElementById("case-card-template");
  target.replaceChildren(
    ...cases.map((caseData) => {
      const fragment = caseTemplate.content.cloneNode(true);
      fragment.querySelector(".case-title").textContent = caseLabel(caseData.case);
      const formula = fragment.querySelector(".case-formula");
      const details = [caseData.formula, caseData.n_obs ? `${caseData.n_obs} obs` : null]
        .filter(Boolean)
        .join(" · ");
      formula.textContent = details;
      const metric = caseData[metricKey];
      const fastest = fragment.querySelector(".case-fastest");
      if (!metric) {
        fastest.textContent =
          metricKey === "fit_prepared"
            ? "Prepared fit is LMM-only; this case has cold-fit timings."
            : "No successful measurements";
        return fragment;
      }
      const ratio = metric.rust_over_julia_median;
      const badge = document.createElement("span");
      const qualified = metric.eligible_for_speed_claim === true;
      badge.className = `evidence-badge${qualified ? " verified" : ""}`;
      badge.textContent = qualified ? "Numerically qualified" : metricKey === "fit_prepared" ? "Different timing boundaries" : "Not qualified for a speed claim";
      fastest.before(badge);
      const targetNote = metric.meets_target == null ? "target unresolved" :
        metric.meets_target ? "meets target" : "misses target";
      if (metric.eligible_for_speed_claim === false) {
        const reasons = metric.fit_agreement?.reasons || [];
        fastest.textContent = `${formatRatio(ratio)} Julia · ${metricKey === "fit_prepared" ? "Reuses a prepared Rust design; Julia constructs a new model." : reasons.join("; ") || "Fit agreement or sampling is not established."}`;
      } else if (metric.ratio_interval_95) {
        const [low, high] = metric.ratio_interval_95;
        fastest.textContent = `${formatRatio(ratio)} Julia · 95% interval ${formatRatio(low)}–${formatRatio(high)} · ${metric.faster_implementation} · ${targetNote}`;
      } else {
        fastest.textContent = `${formatVsJulia(ratio)} · ${targetNote}`;
      }
      fillBars(fragment.querySelector(".bars"), metric.entries || [], (entry) => {
        if (entry.implementation === "rust" && ratio != null) {
          return `${formatSeconds(entry.median_seconds)} (${formatRatio(ratio)})`;
        }
        return formatSeconds(entry.median_seconds);
      });
      return fragment;
    }),
  );
}

function renderFairCases(data, metricKey) {
  const query = document.getElementById("case-search").value.trim().toLowerCase();
  const all = data.fair?.cases || [];
  const cases = all.filter(item => `${item.case} ${caseLabel(item.case)} ${item.formula}`.toLowerCase().includes(query));
  document.getElementById("case-count").textContent = `${cases.length} of ${all.length} cases`;
  renderFairCaseCards(
    document.getElementById("case-grid"),
    cases,
    metricKey,
  );
}

function renderOptimizer(data) {
  const report = data.optimizer_comparison;
  const section = document.getElementById("optimizer-section");
  if (!report?.cases?.length) return;
  section.hidden = false;
  document.getElementById("optimizer-decision").textContent = report.decision_text || "Argmin remains the default. Basin is available as an optional backend.";
  const protocol = report.protocol || {};
  document.getElementById("optimizer-protocol").textContent = `${protocol.processes_per_backend} processes per backend · ${protocol.samples_per_case_backend_metric} measured fits per case and backend · alternating order · ${protocol.threads} thread. ${report.caveat || ""}`;
  const metricKey = document.getElementById("optimizer-metric").value;
  const cases = report.cases.filter(item => item[metricKey]);
  document.getElementById("optimizer-caption").textContent = metricKey === "cold_fit"
    ? "Complete fits on loaded data; medians and paired block ratios."
    : "Both Rust backends reuse a prepared design; medians and paired block ratios.";
  document.getElementById("optimizer-rows").replaceChildren(...cases.map(item => {
    const row = document.createElement("tr");
    const metric = item[metricKey];
    const cells = [caseLabel(item.case), formatSeconds(metric.argmin_seconds), formatSeconds(metric.basin_seconds), formatRatio(metric.basin_over_argmin), !item.fit_agreement ? "Fit mismatch" : metric.competitive ? "Within 5% margin" : "Competitiveness not established"];
    cells.forEach((text, index) => {
      const td = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) td.scope = "row";
      td.textContent = text;
      if (index === 3) {
        const interval = document.createElement("small");
        interval.textContent = `${formatRatio(metric.ratio_interval_95[0])}–${formatRatio(metric.ratio_interval_95[1])}`;
        td.append(interval);
      }
      row.append(td);
    });
    return row;
  }));
}

function renderCiFair(data) {
  const section = document.getElementById("ci-fair-section");
  const grid = document.getElementById("ci-fair-grid");
  if (!data.ci_fair || !(data.ci_fair.cases || []).length) {
    section.hidden = true;
    grid.replaceChildren();
    return;
  }
  section.hidden = false;
  const header = section.querySelector("p");
  const cfg = data.ci_fair.config || {};
  header.textContent = `${data.ci_fair.label}: ${cfg.warmups ?? "?"} warmup(s), ${cfg.repeats ?? "?"} repeats on the GitHub-hosted runner. Not the workstation completion baseline.`;
  renderFairCaseCards(grid, data.ci_fair.cases, "cold_fit");
}

function renderExternal(data) {
  const section = document.getElementById("external-section");
  const grid = document.getElementById("external-grid");
  const families = data.external?.families || [];
  if (!families.length) {
    section.hidden = true;
    grid.replaceChildren();
    return;
  }
  section.hidden = false;
  const caseTemplate = document.getElementById("case-card-template");
  const blocks = [];
  families.forEach((family) => {
    const heading = document.createElement("h3");
    heading.className = "family-title";
    heading.textContent = family.label || family.family;
    blocks.push(heading);
    const familyGrid = document.createElement("div");
    familyGrid.className = "case-grid";
    family.cases.forEach((caseData) => {
      const fragment = caseTemplate.content.cloneNode(true);
      fragment.querySelector(".case-title").textContent = caseLabel(caseData.case);
      fragment.querySelector(".case-formula").textContent = caseData.formula || "";
      const fastest = (caseData.entries || []).find((entry) => entry.is_fastest);
      fragment.querySelector(".case-fastest").textContent = fastest
        ? `Fastest: ${fastest.label || fastest.implementation}`
        : "";
      fillBars(fragment.querySelector(".bars"), caseData.entries || [], (entry) =>
        formatSeconds(entry.median_seconds),
      );
      familyGrid.append(fragment);
    });
    blocks.push(familyGrid);
  });
  grid.replaceChildren(...blocks);
}

function renderCrossLanguage(data) {
  const section = document.getElementById("cross-language-section");
  const payload = data.cross_language;
  if (!payload || !(payload.cases || []).length) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  const caveat = document.getElementById("cross-language-caveat");
  if (payload.caveat) {
    caveat.textContent = payload.caveat;
  }

  const summary = document.getElementById("cross-ratio-summary");
  const nonRust = (payload.implementation_summary || []).filter(
    (item) => item.implementation !== "rust",
  );
  const maxRatio = Math.max(
    1,
    ...nonRust.map((item) => item.geometric_mean_relative_to_rust || 0),
  );
  summary.replaceChildren(
    ...(payload.implementation_summary || []).map((item) => {
      const row = document.createElement("div");
      row.className = "ratio-row";
      const label = document.createElement("div");
      label.textContent = item.label || item.implementation;
      const track = document.createElement("div");
      track.className = "ratio-track";
      const fill = document.createElement("div");
      fill.className = "ratio-fill";
      const ratio = item.geometric_mean_relative_to_rust || 1;
      fill.style.width = `${Math.max((ratio / maxRatio) * 100, 3)}%`;
      track.append(fill);
      const value = document.createElement("div");
      value.textContent =
        item.implementation === "rust" ? "baseline" : `${formatRatio(ratio)} vs Rust`;
      row.append(label, track, value);
      return row;
    }),
  );

  const caseTemplate = document.getElementById("case-card-template");
  const grid = document.getElementById("cross-case-grid");
  grid.replaceChildren(
    ...(payload.cases || []).map((caseData) => {
      const fragment = caseTemplate.content.cloneNode(true);
      fragment.querySelector(".case-title").textContent = caseLabel(caseData.case);
      fragment.querySelector(".case-formula").textContent = "";
      fragment.querySelector(".case-fastest").textContent = caseData.fastest_implementation
        ? `Fastest in this run: ${caseData.fastest_implementation}`
        : "No successful measurements";
      fillBars(fragment.querySelector(".bars"), caseData.entries || [], (entry) => {
        const suffix =
          entry.relative_to_rust && entry.implementation !== "rust"
            ? ` (${formatRatio(entry.relative_to_rust)} vs Rust)`
            : "";
        return `${formatSeconds(entry.median_seconds)}${suffix}`;
      });
      return fragment;
    }),
  );
}

function renderEnvironment(data) {
  const list = document.getElementById("environment-list");
  const fair = data.fair || {};
  const machine = fair.machine_info || {};
  const versions = fair.runtime_versions || {};
  const entries = [
    ["Fair source", fair.source_path],
    ["Source state", fair.provenance?.working_tree_dirty === true
      ? "Measured with working-tree changes; see the dated report for provenance."
      : fair.provenance?.working_tree_dirty === false ? "Clean checkout at recorded revision" : "Not recorded"],
    ["Platform", machine.platform || machine.system],
    ["Machine", machine.machine || machine.processor],
    ["Processor", machine.cpu_model || machine.processor],
    ["CPU count", machine.cpu_count],
    ["Rust", versions.rustc],
    ["Julia", versions.julia],
    ["Threads", fair.provenance?.thread_environment?.OMP_NUM_THREADS],
  ];
  if (data.external) {
    entries.push(["External source", data.external.source_path]);
    entries.push(["External host", data.external.host?.system || data.external.host?.machine]);
    if ((data.external.skipped || []).length) {
      entries.push(["Skipped", data.external.skipped.join("; ")]);
    }
  }
  if (data.ci_fair) {
    entries.push(["CI fair source", data.ci_fair.source_path]);
    entries.push(["CI platform", data.ci_fair.machine_info?.platform]);
  }

  list.replaceChildren(
    ...entries.flatMap(([term, description]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = description == null || description === "" ? "n/a" : String(description);
      return [dt, dd];
    }),
  );
}

function bindMetricToggle(data) {
  const buttons = document.querySelectorAll(".metric-btn");
  let selected = "cold_fit";
  document.getElementById("case-search").addEventListener("input", () => renderFairCases(data, selected));
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => { item.classList.remove("is-active"); item.setAttribute("aria-pressed", "false"); });
      button.classList.add("is-active");
      button.setAttribute("aria-pressed", "true");
      selected = button.dataset.metric;
      renderFairCases(data, selected);
    });
  });
}

async function main() {
  const response = await fetch("./data/latest.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load benchmark data: ${response.status}`);
  }
  const data = await response.json();
  if (data.schema_version !== 2) {
    throw new Error("Dashboard data is an old schema. Rebuild with scripts/build_benchmark_site.py.");
  }
  renderHeroLinks(data);
  renderSummary(data);
  renderRatioSummary(data);
  renderFairCases(data, "cold_fit");
  renderCiFair(data);
  renderExternal(data);
  renderOptimizer(data);
  document.getElementById("optimizer-metric").addEventListener("change", () => renderOptimizer(data));
  renderCrossLanguage(data);
  renderEnvironment(data);
  bindMetricToggle(data);
}

main().catch((error) => {
  const runMeta = document.getElementById("run-meta");
  runMeta.textContent = error.message;
});
