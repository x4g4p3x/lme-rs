function formatSeconds(seconds) {
  if (seconds >= 10) {
    return `${seconds.toFixed(2)} s`;
  }
  if (seconds >= 1) {
    return `${seconds.toFixed(3)} s`;
  }
  return `${(seconds * 1000).toFixed(2)} ms`;
}

function formatRatio(value) {
  if (value == null) {
    return "n/a";
  }
  return `${value.toFixed(2)}x`;
}

function titleCase(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function renderHeroLinks(data) {
  const heroLinks = document.getElementById("hero-links");
  const links = [];

  if (data.run_url) {
    links.push({ href: data.run_url, label: "Workflow run" });
  }
  if (data.release_url) {
    links.push({ href: data.release_url, label: "GitHub release" });
    if (data.asset_urls.cross_language) {
      links.push({
        href: data.asset_urls.cross_language,
        label: "Cross-language JSON",
      });
    }
    if (data.asset_urls.criterion) {
      links.push({
        href: data.asset_urls.criterion,
        label: "Criterion archive",
      });
    }
  }

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
  const generatedAt = new Date(data.generated_at).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
  runMeta.textContent = `Generated ${generatedAt} for ${data.ref_name || data.git_sha.slice(0, 7)}.`;

  const statsGrid = document.getElementById("stats-grid");
  const stats = [
    ["Cases", data.config.cases.length],
    ["Implementations", data.config.implementations.length],
    ["Warmups", data.config.warmups],
    ["Repeats", data.config.repeats],
    ["CPU count", data.machine_info.cpu_count],
    ["Git SHA", data.git_sha.slice(0, 12)],
  ];

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
  const nonRust = data.implementation_summary.filter(
    (item) => item.implementation !== "rust",
  );
  const maxRatio = Math.max(
    ...nonRust.map((item) => item.geometric_mean_relative_to_rust || 0),
    1,
  );

  container.replaceChildren(
    ...data.implementation_summary.map((item) => {
      const row = document.createElement("div");
      row.className = "ratio-row";

      const label = document.createElement("div");
      label.textContent = titleCase(item.implementation);

      const track = document.createElement("div");
      track.className = "ratio-track";
      const fill = document.createElement("div");
      fill.className = "ratio-fill";
      const ratio = item.geometric_mean_relative_to_rust || 1;
      fill.style.width = `${Math.max((ratio / maxRatio) * 100, 3)}%`;
      track.append(fill);

      const value = document.createElement("div");
      if (item.implementation === "rust") {
        value.textContent = "baseline";
      } else {
        value.textContent = `${formatRatio(ratio)} slower`;
      }

      row.append(label, track, value);
      return row;
    }),
  );
}

function renderCases(data) {
  const caseGrid = document.getElementById("case-grid");
  const caseTemplate = document.getElementById("case-card-template");
  const rowTemplate = document.getElementById("bar-row-template");

  caseGrid.replaceChildren(
    ...data.cases.map((caseData) => {
      const fragment = caseTemplate.content.cloneNode(true);
      fragment.querySelector(".case-title").textContent = caseData.case;
      fragment.querySelector(".case-fastest").textContent =
        caseData.fastest_implementation
          ? `Fastest in this run: ${titleCase(caseData.fastest_implementation)}`
          : "No successful measurements";

      const bars = fragment.querySelector(".bars");
      caseData.entries.forEach((entry) => {
        const row = rowTemplate.content.cloneNode(true);
        row.querySelector(".bar-label").textContent = entry.implementation;
        const fill = row.querySelector(".bar-fill");
        fill.classList.add(entry.implementation);
        fill.style.width = `${Math.max(entry.width_fraction * 100, 2)}%`;
        const value = row.querySelector(".bar-value");
        const ratioSuffix =
          entry.relative_to_rust && entry.implementation !== "rust"
            ? ` (${formatRatio(entry.relative_to_rust)} slower)`
            : "";
        value.textContent = `${formatSeconds(entry.median_seconds)}${ratioSuffix}`;
        bars.append(row);
      });

      return fragment;
    }),
  );
}

function renderEnvironment(data) {
  const list = document.getElementById("environment-list");
  const entries = [
    ["Platform", data.machine_info.platform],
    ["Machine", data.machine_info.machine],
    ["Rust", data.runtime_versions.rustc],
    ["Python", data.runtime_versions.python],
    ["R", data.runtime_versions.Rscript],
    ["Julia", data.runtime_versions.julia],
  ];

  list.replaceChildren(
    ...entries.flatMap(([term, description]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = description || "n/a";
      return [dt, dd];
    }),
  );
}

async function main() {
  const response = await fetch("./data/latest.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load benchmark data: ${response.status}`);
  }

  const data = await response.json();
  renderHeroLinks(data);
  renderSummary(data);
  renderRatioSummary(data);
  renderCases(data);
  renderEnvironment(data);
}

main().catch((error) => {
  const runMeta = document.getElementById("run-meta");
  runMeta.textContent = error.message;
});
