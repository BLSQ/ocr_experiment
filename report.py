"""Build an HTML report that compares the preprocessing runs saved by evaluate_preprocessing.py.

Usage (from the project root):
    python report.py

Reads every results/preprocessing/<pipeline>_<time>.json and keeps only what describes the run: the pipeline name,
its steps and settings, the photo folder, the date and the summary (no photo names). Writes them into
results/reports/preprocessing.html, overwriting it. The numbers are built into the page, so it opens straight from
disk; the charts are drawn with Plotly.js, which the page downloads from a CDN when it is opened.

The page shows a sortable table of the runs, the accuracy per pipeline, accuracy against time per photo, the time
per step, and the history of pipelines that were run more than once. A switch at the top shows either the latest
run of each pipeline or every run.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results/preprocessing")
REPORT_PATH = Path("results/reports/preprocessing.html")


def describe_steps(steps: list) -> str:
    """Write the steps of a pipeline on one line, e.g. "page_crop → channel(channel=red) → table_crop".

    Args:
        steps: The "steps" list of a result file: [name, {settings}] pairs.

    Returns:
        The steps as text.
    """
    parts = []
    for name, settings in steps:
        options = ", ".join(f"{key}={value}" for key, value in settings.items())
        parts.append(f"{name}({options})" if options else name)
    return " → ".join(parts)


def load_runs(folder: Path) -> list[dict]:
    """Read the description and summary of every run saved in a folder.

    Args:
        folder: Folder with the JSON files of evaluate_preprocessing.py.

    Returns:
        One dict per run: file name, pipeline, steps, photo folder, date and every value of the summary.
    """
    runs = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "file": path.name,
                    "pipeline": data["pipeline"],
                    "steps": describe_steps(data["steps"]),
                    "photos_dir": data.get("photos_dir", ""),
                    "date": data["date"],
                    **data["summary"],
                }
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:  # skip unreadable files, keep going
            print(f"Skipped {path.name}: {type(exc).__name__}: {exc}")
    return runs


# The page. __RUNS__ and __GENERATED__ are replaced by the data and the time the report was made.
# Colours: a colour-blind-checked palette, with its own steps for dark mode. Each check (grid, rows, ...) and each
# step (page_crop, ...) always gets the same colour, so colours mean the same thing in every chart.
TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preprocessing report</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.1.0/plotly.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --border: rgba(11, 11, 11, 0.10);
    --text: #0b0b0b; --text-2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
    --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --other: #898781;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --border: rgba(255, 255, 255, 0.10);
      --text: #f0efec; --text-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
      --series-5: #d55181; --series-6: #008300; --series-7: #9085e9;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --border: rgba(255, 255, 255, 0.10);
    --text: #f0efec; --text-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9;
  }
  body {
    margin: 0; background: var(--page); color: var(--text);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 0 0 4px; }
  .sub, .note { color: var(--text-2); margin: 0 0 12px; }
  .note { font-size: 13px; margin: 6px 0 0; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; margin: 16px 0; }
  .controls label { color: var(--text-2); }
  select { font: inherit; }
  section {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin: 0 0 16px;
  }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  td .steps { white-space: normal; color: var(--muted); font-size: 12px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  th { color: var(--text-2); font-weight: 600; cursor: pointer; user-select: none; }
  th:hover { color: var(--text); }
  td.missing { color: var(--muted); }
  .chart { width: 100%; }
</style>
</head>
<body>
<main>
  <h1>Preprocessing report</h1>
  <p class="sub">Runs from <code>results/preprocessing/</code>. Made <span id="generated"></span>.</p>

  <div class="controls">
    <label><input type="checkbox" id="show-all"> Show every run (not only the latest of each pipeline)</label>
    <label>Accuracy measure for the charts
      <select id="measure">
        <option value="grid">grid (rows and columns right)</option>
        <option value="rows" selected>rows</option>
        <option value="cols">columns</option>
        <option value="turn">turn</option>
      </select>
    </label>
  </div>

  <section>
    <h2>Runs</h2>
    <p class="sub">Click a column title to sort. "x/y" = photos right out of photos checked.</p>
    <div class="table-wrap"><table id="runs"></table></div>
  </section>

  <section>
    <h2>Accuracy per pipeline</h2>
    <p class="sub">Share of checked photos where each value was right.</p>
    <div id="accuracy" class="chart"></div>
    <p class="note" id="accuracy-note"></p>
  </section>

  <section>
    <h2>Accuracy against time</h2>
    <p class="sub">Up and to the left is better: more photos right, in less time.</p>
    <div id="tradeoff" class="chart"></div>
  </section>

  <section>
    <h2>Time per step</h2>
    <p class="sub">Mean seconds per photo spent in each step.</p>
    <div id="steps" class="chart"></div>
  </section>

  <section id="history-section">
    <h2>History</h2>
    <p class="sub">Every run of one pipeline, oldest first.
      <select id="history-pipeline"></select></p>
    <div id="history" class="chart"></div>
  </section>
</main>

<script>
const RUNS = __RUNS__;
document.getElementById("generated").textContent = "__GENERATED__";

const CHECKS = ["grid", "rows", "cols", "turn"];
// Steps in pipeline order; each keeps its colour slot. Unknown steps share the "other" grey.
const STEP_ORDER = ["page_crop", "deskew", "orientation", "channel", "deblur", "table_crop", "measure_grid"];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const checkColour = (check) => css(`--series-${CHECKS.indexOf(check) + 1}`);
const stepColour = (step) => {
  const slot = STEP_ORDER.indexOf(step);
  return slot >= 0 ? css(`--series-${slot + 1}`) : css("--other");
};

const state = { showAll: false, measure: "rows", sortKey: "date", sortUp: false };

// --- Data helpers ---

const shortDate = (iso) => iso.replace("T", " ").slice(0, 16);
const label = (run) => (state.showAll ? `${run.pipeline} · ${shortDate(run.date)}` : run.pipeline);
const checked = (run, check) => run[`${check}_checked`] > 0 && run[`${check}_correct`] !== null;
const nRight = (run, check) => Math.round(run[`${check}_correct`] * run[`${check}_checked`]);
const fraction = (run, check) =>
  checked(run, check) ? `${nRight(run, check)}/${run[`${check}_checked`]}` : "not checked";

function visibleRuns() {
  const byDate = [...RUNS].sort((a, b) => a.date.localeCompare(b.date));
  if (state.showAll) return byDate;
  const latest = new Map();
  for (const run of byDate) latest.set(run.pipeline, run);  // later runs replace earlier ones
  return [...latest.values()];
}

// Best first by the chosen measure, then by the other checks; not checked counts as worst.
function rankRuns(runs) {
  const score = (run, check) => (checked(run, check) ? run[`${check}_correct`] : -1);
  const order = [state.measure, ...CHECKS.filter((c) => c !== state.measure)];
  return [...runs].sort((a, b) => {
    for (const check of order) {
      const diff = score(b, check) - score(a, check);
      if (diff !== 0) return diff;
    }
    return a.mean_seconds_per_photo - b.mean_seconds_per_photo;
  });
}

// --- Plotly look, read from the CSS colours so it follows light and dark mode ---

function baseLayout(extra) {
  const axis = { gridcolor: css("--grid"), linecolor: css("--axis"), zerolinecolor: css("--axis"),
                 tickfont: { color: css("--muted") }, automargin: true };
  return {
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css("--text-2"), size: 12 },
    margin: { l: 10, r: 20, t: 30, b: 40 },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--border"), font: { color: css("--text") } },
    legend: { orientation: "h", x: 0, y: 1, yanchor: "bottom", traceorder: "normal" },  // above the plot
    xaxis: { ...axis, ...(extra.xaxis || {}) },
    yaxis: { ...axis, ...(extra.yaxis || {}) },
    ...Object.fromEntries(Object.entries(extra).filter(([key]) => !["xaxis", "yaxis"].includes(key))),
  };
}
const CONFIG = { displaylogo: false, responsive: true };
const barHeight = (n, perRow) => 110 + n * perRow;
const emptyNote = (text) => [{ text, showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5,
                               font: { color: css("--muted") } }];

// --- Table ---

function columns() {
  const checkCol = (check) => ({
    key: check, title: check, num: true,
    sortValue: (r) => (checked(r, check) ? r[`${check}_correct`] : -1),
    text: (r) => (checked(r, check) ? `${fraction(r, check)} (${Math.round(r[`${check}_correct`] * 100)}%)` : "-"),
  });
  return [
    { key: "pipeline", title: "pipeline and steps", sortValue: (r) => r.pipeline, text: (r) => r.pipeline,
      sub: (r) => r.steps },
    { key: "date", title: "date", sortValue: (r) => r.date, text: (r) => shortDate(r.date) },
    ...CHECKS.map(checkCol),
    { key: "failed", title: "failed", num: true, sortValue: (r) => r.failed_photos, text: (r) => r.failed_photos },
    { key: "seconds", title: "s/photo", num: true, sortValue: (r) => r.mean_seconds_per_photo,
      text: (r) => r.mean_seconds_per_photo.toFixed(2) },
    { key: "photos", title: "photos", sortValue: (r) => r.photos_dir, text: (r) => `${r.photos} in ${r.photos_dir}` },
  ];
}

function drawTable(runs) {
  const cols = columns();
  const col = cols.find((c) => c.key === state.sortKey);
  const rows = [...runs].sort((a, b) => {
    const [x, y] = [col.sortValue(a), col.sortValue(b)];
    const cmp = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
    return state.sortUp ? cmp : -cmp;
  });

  const table = document.getElementById("runs");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const c of cols) {
    const th = document.createElement("th");
    th.textContent = c.title + (c.key === state.sortKey ? (state.sortUp ? " ▲" : " ▼") : "");
    if (c.num) th.className = "num";
    th.addEventListener("click", () => {
      state.sortUp = state.sortKey === c.key ? !state.sortUp : false;
      state.sortKey = c.key;
      drawTable(visibleRuns());
    });
    head.appendChild(th);
  }
  const body = table.createTBody();
  for (const run of rows) {
    const tr = body.insertRow();
    for (const c of cols) {
      const td = tr.insertCell();
      td.textContent = c.text(run);
      td.className = [c.num ? "num" : "", td.textContent === "-" ? "missing" : ""].join(" ").trim();
      if (c.sub) {
        const sub = document.createElement("div");
        sub.className = "steps";
        sub.textContent = c.sub(run);
        td.appendChild(sub);
      }
    }
  }
}

// --- Charts ---

function drawAccuracy(runs) {
  const ranked = rankRuns(runs);
  const used = CHECKS.filter((check) => ranked.some((run) => checked(run, check)));
  const unused = CHECKS.filter((check) => !used.includes(check));
  const traces = used.map((check) => ({
    type: "bar", orientation: "h", name: check,
    y: ranked.map(label),
    x: ranked.map((run) => (checked(run, check) ? run[`${check}_correct`] * 100 : null)),
    text: ranked.map((run) => fraction(run, check)),
    textposition: "outside", textfont: { color: css("--muted"), size: 11 }, cliponaxis: false,
    marker: { color: checkColour(check), line: { color: css("--surface"), width: 2 } },
    hovertemplate: `<b>%{text}</b> ${check} right<extra>%{y}</extra>`,
  }));
  const layout = baseLayout({
    barmode: "group", bargap: 0.25, barcornerradius: 4, height: barHeight(ranked.length, used.length * 16 + 18),
    xaxis: { range: [0, 110], ticksuffix: "%", title: { text: "photos right" } },
    yaxis: { autorange: "reversed" },  // best pipeline on top, bars in legend order
  });
  Plotly.react("accuracy", traces, layout, CONFIG);
  document.getElementById("accuracy-note").textContent = unused.length
    ? `Not checked in any shown run: ${unused.join(", ")} (fill it in expected.csv).` : "";
}

function drawTradeoff(runs) {
  const measure = state.measure;
  const points = runs.filter((run) => checked(run, measure));
  const trace = {
    type: "scatter", mode: "markers+text",
    x: points.map((run) => run.mean_seconds_per_photo),
    y: points.map((run) => run[`${measure}_correct`] * 100),
    text: points.map(label), textfont: { color: css("--text-2"), size: 11 },
    // Labels alternate above and below, so pipelines with a similar time and accuracy don't overlap as much.
    textposition: points.map((_, i) => (i % 2 ? "bottom center" : "top center")),
    customdata: points.map((run) => fraction(run, measure)),
    marker: { size: 11, color: checkColour(measure), line: { color: css("--surface"), width: 2 } },
    hovertemplate: `<b>%{customdata}</b> ${measure} right · %{x:.2f} s/photo<extra>%{text}</extra>`,
  };
  const layout = baseLayout({
    height: 360, showlegend: false,
    // Room on the right so the label of the slowest pipeline isn't cut off.
    xaxis: { title: { text: "seconds per photo" },
             range: [0, 1.25 * Math.max(1, ...points.map((run) => run.mean_seconds_per_photo))] },
    yaxis: { title: { text: `${measure} right` }, range: [-5, 110], ticksuffix: "%" },
    annotations: points.length ? [] : emptyNote(`${measure} was not checked in any shown run`),
  });
  Plotly.react("tradeoff", [trace], layout, CONFIG);
}

function drawSteps(runs) {
  const order = rankRuns(runs);  // same order as the accuracy chart
  const seen = new Set(runs.flatMap((run) => Object.keys(run.mean_seconds_per_step || {})));
  const steps = [...STEP_ORDER.filter((s) => seen.has(s)), ...[...seen].filter((s) => !STEP_ORDER.includes(s))];
  const traces = steps.map((step) => ({
    type: "bar", orientation: "h", name: step,
    y: order.map(label),
    x: order.map((run) => (run.mean_seconds_per_step || {})[step] ?? null),
    marker: { color: stepColour(step), line: { color: css("--surface"), width: 2 } },
    hovertemplate: `<b>%{x:.3f} s</b> ${step}<extra>%{y}</extra>`,
  }));
  const layout = baseLayout({
    barmode: "stack", bargap: 0.35, barcornerradius: 4, height: barHeight(order.length, 34),
    xaxis: { title: { text: "seconds per photo" } },
    yaxis: { autorange: "reversed" },
  });
  Plotly.react("steps", traces, layout, CONFIG);
}

function drawHistory() {
  const counts = {};
  for (const run of RUNS) counts[run.pipeline] = (counts[run.pipeline] || 0) + 1;
  const repeated = Object.keys(counts).filter((p) => counts[p] > 1).sort();
  const section = document.getElementById("history-section");
  section.hidden = repeated.length === 0;
  if (!repeated.length) return;

  const select = document.getElementById("history-pipeline");
  if (select.options.length !== repeated.length) {
    select.replaceChildren(...repeated.map((p) => new Option(`${p} (${counts[p]} runs)`, p)));
  }
  const runs = RUNS.filter((run) => run.pipeline === select.value).sort((a, b) => a.date.localeCompare(b.date));
  const traces = CHECKS.filter((check) => runs.some((run) => checked(run, check))).map((check) => ({
    type: "scatter", mode: "lines+markers", name: check,
    x: runs.map((run) => shortDate(run.date)),
    y: runs.map((run) => (checked(run, check) ? run[`${check}_correct`] * 100 : null)),
    customdata: runs.map((run) => fraction(run, check)),
    line: { color: checkColour(check), width: 2 },
    marker: { size: 8, color: checkColour(check), line: { color: css("--surface"), width: 2 } },
    hovertemplate: `<b>%{customdata}</b> ${check}<extra></extra>`,
  }));
  const layout = baseLayout({
    height: 320, hovermode: "x unified",
    xaxis: { type: "category", title: { text: "run" } },
    yaxis: { range: [-5, 105], ticksuffix: "%", title: { text: "photos right" } },
    annotations: traces.length ? [] : emptyNote("Nothing was checked in these runs (expected.csv was empty)"),
  });
  Plotly.react("history", traces, layout, CONFIG);
}

function drawAll() {
  const runs = visibleRuns();
  drawTable(runs);
  drawAccuracy(runs);
  drawTradeoff(runs);
  drawSteps(runs);
  drawHistory();
}

document.getElementById("show-all").addEventListener("change", (e) => { state.showAll = e.target.checked; drawAll(); });
document.getElementById("measure").addEventListener("change", (e) => { state.measure = e.target.value; drawAll(); });
document.getElementById("history-pipeline").addEventListener("change", drawHistory);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
drawAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    runs = load_runs(RESULTS_DIR)
    if not runs:
        sys.exit(f"No result files in {RESULTS_DIR}/. Run evaluate_preprocessing.py first.")

    # "</" is escaped so a value can never close the <script> tag early.
    data = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__RUNS__", data).replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M"))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Read {len(runs)} runs of {len({run['pipeline'] for run in runs})} pipelines. Saved {REPORT_PATH}")
