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

function addLink(links, href, label) {
  if (!href) {
    return;
  }
  links.push({ href, label });
}

function renderHeroLinks(data) {
  const heroLinks = document.getElementById("hero-links");
  const links = [];
  const siteLinks = data.links || {};

  addLink(links, siteLinks.coverage_url, "Coverage map");
  addLink(links, siteLinks.methodology_url, "Methodology");
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
  runMeta.textContent = `${fair.label || "Fair harness"} · ${generatedAt} · ${sha}`;

  const headline = data.headline || {};
  const config = fair.config || {};
  const stats = [
    ["Cold-fit target", `${headline.cold_fit_passes}/${headline.cold_fit_cases} under ${formatRatio(headline.cold_fit_target)}`],
    ["Geomean Rust/Julia", formatVsJulia(headline.geometric_mean_rust_over_julia_cold_fit)],
    ["Median Rust/Julia", formatVsJulia(headline.median_rust_over_julia_cold_fit)],
    ["Warmups / repeats", `${config.warmups ?? "n/a"} / ${config.repeats ?? "n/a"}`],
    ["Cases", (fair.cases || []).length],
    ["Git SHA", sha],
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
    ["Cold fit geomean", summary.geometric_mean_rust_over_julia_cold_fit],
    ["Cold fit median", summary.median_rust_over_julia_cold_fit],
    ["Prepared geomean", summary.geometric_mean_rust_over_julia_prepared],
    ["Prepared median", summary.median_rust_over_julia_prepared],
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
      value.textContent = formatVsJulia(ratio);
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
      fragment.querySelector(".case-title").textContent = caseData.case;
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
      const targetNote = metric.meets_target ? "meets target" : "misses target";
      fastest.textContent = `${formatVsJulia(ratio)} · ${targetNote}`;
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
  renderFairCaseCards(
    document.getElementById("case-grid"),
    data.fair?.cases || [],
    metricKey,
  );
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
      fragment.querySelector(".case-title").textContent = caseData.case;
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
      fragment.querySelector(".case-title").textContent = caseData.case;
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
    ["Platform", machine.platform || machine.system],
    ["Machine", machine.machine || machine.processor],
    ["CPU count", machine.cpu_count],
    ["Rust", versions.rustc],
    ["Julia", versions.julia],
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
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.classList.remove("is-active"));
      button.classList.add("is-active");
      renderFairCases(data, button.dataset.metric);
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
  renderCrossLanguage(data);
  renderEnvironment(data);
  bindMetricToggle(data);
}

main().catch((error) => {
  const runMeta = document.getElementById("run-meta");
  runMeta.textContent = error.message;
});
