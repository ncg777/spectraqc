from __future__ import annotations

import json
import os
import sys
import webbrowser
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from spectraqc.version import __version__


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _strip_keep_empty(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _uploads_dir() -> Path:
    base = Path.cwd() / ".spectraqc_gui_uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_index_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpectraQC GUI</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --card: #ffffff;
      --text: #1b1f24;
      --muted: #5c6773;
      --border: #e2e7f0;
      --primary: #3b6bed;
      --primary-dark: #2f54c9;
      --danger: #c62828;
      --success: #2e7d32;
      --warn: #ef6c00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      background: #0f172a;
      color: #fff;
      padding: 20px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    header h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.3px;
    }}
    header small {{
      color: #cbd5f5;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px;
      background: #fff;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    nav button {{
      border: 1px solid var(--border);
      background: #f3f5f8;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 13px;
    }}
    nav button.active {{
      background: var(--primary);
      color: #fff;
      border-color: var(--primary);
    }}
    main {{
      padding: 24px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    section {{
      display: none;
    }}
    section.active {{
      display: block;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 18px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    input, select, textarea {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 14px;
      font-family: inherit;
    }}
    textarea {{
      min-height: 120px;
      resize: vertical;
    }}
    button.primary {{
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 10px 16px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.primary:hover {{
      background: var(--primary-dark);
    }}
    button.secondary {{
      background: #f1f4ff;
      border: 1px solid #c7d2fe;
      color: #2f3a8f;
      padding: 10px 14px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 13px;
      white-space: nowrap;
    }}
    button.secondary:hover {{
      background: #e0e7ff;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .result {{
      background: #0b1020;
      color: #e2e8f0;
      padding: 12px;
      border-radius: 10px;
      font-family: "JetBrains Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge.pass {{ background: #e8f5e9; color: var(--success); }}
    .badge.warn {{ background: #fff3e0; color: var(--warn); }}
    .badge.fail {{ background: #ffebee; color: var(--danger); }}
    .badge.error {{ background: #ffebee; color: var(--danger); }}
    .row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .row.tight {{
      flex-wrap: nowrap;
    }}
    .row.tight input {{
      flex: 1 1 auto;
    }}
    .row > * {{
      flex: 1;
    }}
    .summary-list {{
      list-style: none;
      padding: 0;
      margin: 12px 0 0;
    }}
    .summary-list li {{
      padding: 6px 0;
      border-bottom: 1px dashed var(--border);
      font-size: 13px;
    }}
    .pill {{
      display: inline-block;
      background: #eef2ff;
      color: #3f51b5;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      margin-right: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SpectraQC GUI</h1>
      <small>Local control panel for SpectraQC v{__version__}</small>
    </div>
    <div class="muted">Run CLI workflows with a visual interface.</div>
  </header>
  <nav>
    <button class="tab active" data-tab="analyze">Analyze</button>
    <button class="tab" data-tab="validate">Validate</button>
    <button class="tab" data-tab="repair">Repair</button>
    <button class="tab" data-tab="inspect">Inspect Profile</button>
    <button class="tab" data-tab="batch">Batch</button>
    <button class="tab" data-tab="build-profile">Build Profile</button>
    <button class="tab" data-tab="viewer">Report Viewer</button>
  </nav>
  <main>
    <section id="analyze" class="active">
      <div class="card">
        <h2>Analyze Audio</h2>
        <p class="muted">Generate a QCReport for a single audio file.</p>
        <div class="grid">
          <div>
            <label>Audio path</label>
            <input id="analyze-audio" placeholder="/path/to/input.wav">
          </div>
          <div>
            <label>Profile path</label>
            <div class="row tight">
              <input id="analyze-profile" placeholder="/path/to/profile.ref.json">
              <button class="secondary" type="button" data-upload-target="analyze-profile">Upload JSON</button>
            </div>
          </div>
          <div>
            <label>Mode</label>
            <select id="analyze-mode">
              <option value="compliance">compliance</option>
              <option value="exploratory">exploratory</option>
            </select>
          </div>
          <div>
            <label>Output report path (optional)</label>
            <input id="analyze-out" placeholder="/path/to/report.qcreport.json">
          </div>
          <div>
            <label>Repro markdown path (optional)</label>
            <input id="analyze-repro" placeholder="/path/to/report-repro.md">
          </div>
        </div>
        <div class="row" style="margin-top:16px;">
          <button class="primary" id="analyze-run">Run analysis</button>
          <div id="analyze-status" class="muted"></div>
        </div>
        <div id="analyze-result" class="result" style="margin-top:16px; display:none;"></div>
      </div>
    </section>

    <section id="validate">
      <div class="card">
        <h2>Validate Audio</h2>
        <p class="muted">Return a quick PASS/WARN/FAIL decision.</p>
        <div class="grid">
          <div>
            <label>Audio path</label>
            <input id="validate-audio" placeholder="/path/to/input.wav">
          </div>
          <div>
            <label>Profile path</label>
            <div class="row tight">
              <input id="validate-profile" placeholder="/path/to/profile.ref.json">
              <button class="secondary" type="button" data-upload-target="validate-profile">Upload JSON</button>
            </div>
          </div>
          <div>
            <label>Fail on</label>
            <select id="validate-fail-on">
              <option value="fail">fail</option>
              <option value="warn">warn</option>
            </select>
          </div>
        </div>
        <div class="row" style="margin-top:16px;">
          <button class="primary" id="validate-run">Run validation</button>
          <div id="validate-status" class="muted"></div>
        </div>
        <div id="validate-summary" style="margin-top:16px;"></div>
      </div>
    </section>

    <section id="repair">
      <div class="card">
        <h2>Repair Audio</h2>
        <p class="muted">Apply a DSP repair plan and produce a QC report.</p>
        <div class="grid">
          <div>
            <label>Audio path</label>
            <input id="repair-audio" placeholder="/path/to/input.wav">
          </div>
          <div>
            <label>Profile path</label>
            <div class="row tight">
              <input id="repair-profile" placeholder="/path/to/profile.ref.json">
              <button class="secondary" type="button" data-upload-target="repair-profile">Upload JSON</button>
            </div>
          </div>
          <div>
            <label>Repair plan path</label>
            <input id="repair-plan" placeholder="/path/to/repair-plan.yaml">
          </div>
          <div>
            <label>Repair plan JSON (optional)</label>
            <textarea id="repair-plan-json" placeholder='{"steps":[{"name":"dehum","params":{"hum_freq_hz":60}}]}'></textarea>
          </div>
          <div>
            <label>Output audio path (optional)</label>
            <input id="repair-out" placeholder="/path/to/output.repaired.wav">
          </div>
          <div>
            <label>Output report path (optional)</label>
            <input id="repair-report" placeholder="/path/to/report.qcreport.json">
          </div>
        </div>
        <div class="row" style="margin-top:16px;">
          <button class="primary" id="repair-run">Run repair</button>
          <button class="secondary" type="button" id="repair-plan-suggest">Suggest plan</button>
          <div id="repair-status" class="muted"></div>
        </div>
        <div id="repair-result" class="result" style="margin-top:16px; display:none;"></div>
      </div>
    </section>

    <section id="inspect">
      <div class="card">
        <h2>Inspect Reference Profile</h2>
        <p class="muted">Review a reference profile's metadata and thresholds.</p>
        <div class="grid">
          <div>
            <label>Profile path</label>
            <div class="row tight">
              <input id="inspect-profile" placeholder="/path/to/profile.ref.json">
              <button class="secondary" type="button" data-upload-target="inspect-profile">Upload JSON</button>
            </div>
          </div>
        </div>
        <div class="row" style="margin-top:16px;">
          <button class="primary" id="inspect-run">Inspect profile</button>
          <div id="inspect-status" class="muted"></div>
        </div>
        <div id="inspect-summary" style="margin-top:16px;"></div>
      </div>
    </section>

    <section id="batch">
      <div class="card">
        <h2>Batch Analyze</h2>
        <p class="muted">Analyze a folder or manifest and produce batch reports.</p>
        <div class="grid">
          <div>
            <label>Folder (optional)</label>
            <input id="batch-folder" placeholder="/path/to/audio/folder">
          </div>
          <div>
            <label>Manifest (optional)</label>
            <div class="row tight">
              <input id="batch-manifest" placeholder="/path/to/manifest.json">
              <button class="secondary" type="button" data-upload-target="batch-manifest">Upload JSON</button>
            </div>
          </div>
          <div>
            <label>Profile path</label>
            <div class="row tight">
              <input id="batch-profile" placeholder="/path/to/profile.ref.json">
              <button class="secondary" type="button" data-upload-target="batch-profile">Upload JSON</button>
            </div>
          </div>
          <div>
            <label>Mode</label>
            <select id="batch-mode">
              <option value="compliance">compliance</option>
              <option value="exploratory">exploratory</option>
            </select>
          </div>
          <div>
            <label>Output directory</label>
            <input id="batch-out-dir" placeholder="/path/to/output">
          </div>
          <div>
            <label>Workers</label>
            <input id="batch-workers" type="number" min="1" value="4">
          </div>
        </div>
        <div class="row" style="margin-top:12px;">
          <label><input type="checkbox" id="batch-recursive"> Recurse into subfolders</label>
        </div>
        <div class="card" style="margin-top:16px;">
          <h3>Outputs</h3>
          <div class="grid">
            <div>
              <label><input type="checkbox" id="batch-summary-json" checked> Summary JSON</label>
              <input id="batch-summary-json-name" value="batch-summary.json">
            </div>
            <div>
              <label><input type="checkbox" id="batch-summary-md" checked> Summary Markdown</label>
              <input id="batch-summary-md-name" value="batch-summary.md">
            </div>
            <div>
              <label><input type="checkbox" id="batch-kpis-json" checked> KPI JSON</label>
              <input id="batch-kpis-json-name" value="batch-kpis.json">
            </div>
            <div>
              <label><input type="checkbox" id="batch-kpis-csv" checked> KPI CSV</label>
              <input id="batch-kpis-csv-name" value="batch-kpis.csv">
            </div>
            <div>
              <label><input type="checkbox" id="batch-report-md" checked> Report Markdown</label>
              <input id="batch-report-md-name" value="batch-report.md">
            </div>
            <div>
              <label><input type="checkbox" id="batch-report-html" checked> Report HTML</label>
              <input id="batch-report-html-name" value="batch-report.html">
            </div>
            <div>
              <label><input type="checkbox" id="batch-repro-md"> Repro Markdown</label>
              <input id="batch-repro-md-name" value="batch-repro.md">
            </div>
          </div>
        </div>
        <div class="row" style="margin-top:16px;">
          <button class="primary" id="batch-run">Run batch analysis</button>
          <div id="batch-status" class="muted"></div>
        </div>
        <div id="batch-result" style="margin-top:16px;"></div>
      </div>
    </section>

    <section id="build-profile">
      <div class="card">
        <h2>Build Reference Profile</h2>
        <p class="muted">Generate a .ref.json profile from a manifest or folder.</p>
        <div class="grid">
          <div>
            <label>Manifest path (optional)</label>
            <div class="row tight">
              <input id="build-manifest" placeholder="/path/to/manifest.json">
              <button class="secondary" type="button" data-upload-target="build-manifest">Upload JSON</button>
            </div>
          </div>
          <div>
            <label>Folder path (optional)</label>
            <input id="build-folder" placeholder="/path/to/audio/folder">
          </div>
          <div>
            <label>Output path (optional)</label>
            <input id="build-out" placeholder="/path/to/profile.ref.json">
          </div>
          <div>
            <label>Profile name</label>
            <input id="build-name" value="streaming_generic_v1">
          </div>
          <div>
            <label>Profile kind</label>
            <select id="build-kind">
              <option value="streaming">streaming</option>
              <option value="broadcast">broadcast</option>
              <option value="archive">archive</option>
              <option value="custom">custom</option>
            </select>
          </div>
        </div>
        <div class="row" style="margin-top:12px;">
          <label><input type="checkbox" id="build-recursive"> Recurse into subfolders</label>
        </div>
        <div class="row" style="margin-top:16px;">
          <button class="primary" id="build-run">Build profile</button>
          <div id="build-status" class="muted"></div>
        </div>
        <div id="build-result" class="result" style="margin-top:16px; display:none;"></div>
      </div>
    </section>

    <section id="viewer">
      <div class="card">
        <h2>QCReport Viewer</h2>
        <p class="muted">Drop one or more QCReport JSON files to inspect key decisions.</p>
        <input id="viewer-files" type="file" multiple accept=".json">
        <div id="viewer-list" style="margin-top:16px;"></div>
      </div>
    </section>
  </main>
  <script>
    const tabs = document.querySelectorAll(".tab");
    const sections = document.querySelectorAll("main section");
    tabs.forEach((tab) => {{
      tab.addEventListener("click", () => {{
        tabs.forEach((t) => t.classList.remove("active"));
        sections.forEach((s) => s.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).classList.add("active");
      }});
    }});

    const postJSON = async (endpoint, payload) => {{
      const res = await fetch(endpoint, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }});
      const data = await res.json();
      if (!res.ok || !data.ok) {{
        const message = data.error || res.statusText;
        throw new Error(message);
      }}
      return data.data;
    }};

    const uploadJsonForTarget = async (targetId) => {{
      const picker = document.createElement("input");
      picker.type = "file";
      picker.accept = ".json";
      picker.addEventListener("change", async () => {{
        const file = picker.files[0];
        if (!file) return;
        try {{
          const content = await file.text();
          const data = await postJSON("/api/upload-json", {{
            filename: file.name,
            content
          }});
          const target = document.getElementById(targetId);
          if (target) {{
            target.value = data.path;
          }}
        }} catch (err) {{
          alert(`Upload failed: ${err.message}`);
        }}
      }});
      picker.click();
    }};

    const setResult = (element, content) => {{
      element.textContent = content;
      element.style.display = "block";
    }};

    const setStatus = (element, message, isError=false) => {{
      element.textContent = message;
      element.style.color = isError ? "var(--danger)" : "var(--muted)";
    }};

    document.querySelectorAll("[data-upload-target]").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        const target = btn.dataset.uploadTarget;
        uploadJsonForTarget(target);
      }});
    }});

    document.getElementById("analyze-run").addEventListener("click", async () => {{
      const statusEl = document.getElementById("analyze-status");
      const resultEl = document.getElementById("analyze-result");
      setStatus(statusEl, "Running...");
      try {{
        const data = await postJSON("/api/analyze", {{
          audio_path: document.getElementById("analyze-audio").value,
          profile_path: document.getElementById("analyze-profile").value,
          mode: document.getElementById("analyze-mode").value,
          out_path: document.getElementById("analyze-out").value,
          repro_md: document.getElementById("analyze-repro").value
        }});
        setStatus(statusEl, `Status: ${{data.status.toUpperCase()}}`);
        setResult(resultEl, JSON.stringify(data, null, 2));
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    document.getElementById("validate-run").addEventListener("click", async () => {{
      const statusEl = document.getElementById("validate-status");
      const summaryEl = document.getElementById("validate-summary");
      setStatus(statusEl, "Running...");
      summaryEl.innerHTML = "";
      try {{
        const data = await postJSON("/api/validate", {{
          audio_path: document.getElementById("validate-audio").value,
          profile_path: document.getElementById("validate-profile").value,
          fail_on: document.getElementById("validate-fail-on").value
        }});
        setStatus(statusEl, `Status: ${{data.status.toUpperCase()}}`);
        const badge = `<span class="badge ${{data.status}}">${{data.status.toUpperCase()}}</span>`;
        summaryEl.innerHTML = `<div class="row">${{badge}}<div class="muted">Exit code: ${{data.exit_code}}</div></div>`;
        if (data.summary_lines && data.summary_lines.length) {{
          summaryEl.innerHTML += `<ul class="summary-list">${{data.summary_lines.map((l) => `<li>${{l}}</li>`).join("")}}</ul>`;
        }}
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    document.getElementById("repair-run").addEventListener("click", async () => {{
      const statusEl = document.getElementById("repair-status");
      const resultEl = document.getElementById("repair-result");
      setStatus(statusEl, "Running...");
      try {{
        const data = await postJSON("/api/repair", {{
          audio_path: document.getElementById("repair-audio").value,
          profile_path: document.getElementById("repair-profile").value,
          repair_plan: document.getElementById("repair-plan").value,
          plan_json: document.getElementById("repair-plan-json").value,
          out_path: document.getElementById("repair-out").value,
          report_path: document.getElementById("repair-report").value
        }});
        setStatus(statusEl, `Status: ${{data.status.toUpperCase()}}`);
        setResult(resultEl, JSON.stringify(data, null, 2));
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    document.getElementById("repair-plan-suggest").addEventListener("click", async () => {{
      const statusEl = document.getElementById("repair-status");
      const planEl = document.getElementById("repair-plan-json");
      setStatus(statusEl, "Generating plan...");
      try {{
        const data = await postJSON("/api/repair-plan", {{
          audio_path: document.getElementById("repair-audio").value,
          profile_path: document.getElementById("repair-profile").value
        }});
        planEl.value = JSON.stringify(data.plan, null, 2);
        setStatus(statusEl, `Plan ready (${data.summary.suggested_step_count} steps)`);
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    document.getElementById("inspect-run").addEventListener("click", async () => {{
      const statusEl = document.getElementById("inspect-status");
      const summaryEl = document.getElementById("inspect-summary");
      setStatus(statusEl, "Running...");
      summaryEl.innerHTML = "";
      try {{
        const data = await postJSON("/api/inspect", {{
          profile_path: document.getElementById("inspect-profile").value
        }});
        setStatus(statusEl, "Loaded");
        const profile = data.profile;
        summaryEl.innerHTML = `
          <div class="row">
            <span class="pill">${{profile.name}}</span>
            <span class="pill">${{profile.kind}}</span>
            <span class="pill">v${{profile.version}}</span>
          </div>
          <ul class="summary-list">
            <li><strong>Hash:</strong> ${{profile.hash}}</li>
            <li><strong>Bands:</strong> ${{profile.band_count}}</li>
            <li><strong>Frequency grid:</strong> ${{profile.grid_bins}} bins (${{profile.freq_min}} - ${{profile.freq_max}} Hz)</li>
            <li><strong>Band mean thresholds:</strong> pass ${{profile.band_mean_pass}} dB, warn ${{profile.band_mean_warn}} dB</li>
            <li><strong>Band max thresholds:</strong> pass ${{profile.band_max_pass}} dB, warn ${{profile.band_max_warn}} dB</li>
            <li><strong>Tilt thresholds:</strong> pass ${{profile.tilt_pass}} dB/oct, warn ${{profile.tilt_warn}} dB/oct</li>
          </ul>`;
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    document.getElementById("batch-run").addEventListener("click", async () => {{
      const statusEl = document.getElementById("batch-status");
      const resultEl = document.getElementById("batch-result");
      setStatus(statusEl, "Running...");
      resultEl.innerHTML = "";
      const payload = {{
        folder: document.getElementById("batch-folder").value,
        manifest: document.getElementById("batch-manifest").value,
        profile_path: document.getElementById("batch-profile").value,
        mode: document.getElementById("batch-mode").value,
        out_dir: document.getElementById("batch-out-dir").value,
        recursive: document.getElementById("batch-recursive").checked,
        workers: document.getElementById("batch-workers").value,
        summary_json: document.getElementById("batch-summary-json").checked ? document.getElementById("batch-summary-json-name").value : null,
        summary_md: document.getElementById("batch-summary-md").checked ? document.getElementById("batch-summary-md-name").value : null,
        summary_kpis_json: document.getElementById("batch-kpis-json").checked ? document.getElementById("batch-kpis-json-name").value : null,
        summary_kpis_csv: document.getElementById("batch-kpis-csv").checked ? document.getElementById("batch-kpis-csv-name").value : null,
        report_md: document.getElementById("batch-report-md").checked ? document.getElementById("batch-report-md-name").value : null,
        report_html: document.getElementById("batch-report-html").checked ? document.getElementById("batch-report-html-name").value : null,
        repro_md: document.getElementById("batch-repro-md").checked ? document.getElementById("batch-repro-md-name").value : null
      }};
      try {{
        const data = await postJSON("/api/batch", payload);
        setStatus(statusEl, `Completed: ${{data.total_files}} files`);
        const outputs = data.outputs || {{}};
        resultEl.innerHTML = `
          <div class="row">
            <span class="badge ${{data.status}}">${{data.status.toUpperCase()}}</span>
            <div class="muted">Failures: ${{data.failures}}</div>
          </div>
          <div class="result" style="margin-top:12px;">${{JSON.stringify(data.summary, null, 2)}}</div>
          <ul class="summary-list">
            ${{Object.entries(outputs).map(([key, value]) => value ? `<li><strong>${{key}}:</strong> ${{value}}</li>` : "").join("")}}
          </ul>`;
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    document.getElementById("build-run").addEventListener("click", async () => {{
      const statusEl = document.getElementById("build-status");
      const resultEl = document.getElementById("build-result");
      setStatus(statusEl, "Running...");
      try {{
        const data = await postJSON("/api/build-profile", {{
          manifest_path: document.getElementById("build-manifest").value,
          folder_path: document.getElementById("build-folder").value,
          out_path: document.getElementById("build-out").value,
          profile_name: document.getElementById("build-name").value,
          profile_kind: document.getElementById("build-kind").value,
          recursive: document.getElementById("build-recursive").checked
        }});
        setStatus(statusEl, "Profile created");
        setResult(resultEl, JSON.stringify(data, null, 2));
      }} catch (err) {{
        setStatus(statusEl, err.message, true);
      }}
    }});

    const viewerInput = document.getElementById("viewer-files");
    const viewerList = document.getElementById("viewer-list");
    viewerInput.addEventListener("change", async (event) => {{
      const files = Array.from(event.target.files || []);
      viewerList.innerHTML = "";
      for (const file of files) {{
        const text = await file.text();
        try {{
          const report = JSON.parse(text);
          const status = report.decisions?.overall_status || "unknown";
          const confidence = report.confidence?.status || "unknown";
          const reasons = report.confidence?.reasons || [];
          const details = `
            <div class="card" style="margin-top:12px;">
              <div class="row">
                <span class="pill">${{file.name}}</span>
                <span class="badge ${{status}}">${{status.toUpperCase()}}</span>
                <span class="muted">Confidence: ${{confidence}}</span>
              </div>
              <ul class="summary-list">
                ${{(reasons.length ? reasons : ["no confidence warnings"]).map((r) => `<li>${{r}}</li>`).join("")}}
              </ul>
            </div>`;
          viewerList.innerHTML += details;
        }} catch (err) {{
          viewerList.innerHTML += `<div class="card" style="margin-top:12px;"><strong>${{file.name}}</strong>: invalid JSON</div>`;
        }}
      }}
    }});
  </script>
</body>
</html>
"""


class SpectraQCGUIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = _build_index_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        routes = {
            "/api/analyze": self._handle_analyze,
            "/api/validate": self._handle_validate,
            "/api/repair": self._handle_repair,
            "/api/repair-plan": self._handle_repair_plan,
            "/api/inspect": self._handle_inspect,
            "/api/batch": self._handle_batch,
            "/api/build-profile": self._handle_build_profile,
            "/api/upload-json": self._handle_upload_json,
        }
        handler = routes.get(self.path)
        if not handler:
            self.send_error(404)
            return
        try:
            payload = self._json_body()
            data = handler(payload)
            self._send_json(200, {"ok": True, "data": data})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    def _handle_analyze(self, payload: dict) -> dict:
        from spectraqc.cli import main as cli_main

        audio_path = _strip_or_none(payload.get("audio_path"))
        profile_path = _strip_or_none(payload.get("profile_path"))
        if not audio_path or not profile_path:
            raise ValueError("audio_path and profile_path are required.")
        mode = _strip_or_none(payload.get("mode")) or "compliance"
        out_path = _strip_or_none(payload.get("out_path"))
        repro_md = _strip_or_none(payload.get("repro_md"))

        qcreport, decision, profile, algo_ids = cli_main._analyze_audio(
            audio_path,
            profile_path,
            mode=mode,
        )
        output_json = json.dumps(qcreport, indent=2)
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(output_json, encoding="utf-8")
        if repro_md:
            Path(repro_md).parent.mkdir(parents=True, exist_ok=True)
            Path(repro_md).write_text(
                cli_main._render_repro_doc(
                    profile=profile,
                    algorithm_ids=algo_ids,
                    profile_path=profile_path,
                    mode=mode,
                    audio_path=audio_path,
                ),
                encoding="utf-8",
            )
        return {
            "status": decision.overall_status.value,
            "report": qcreport,
            "report_path": out_path,
            "repro_md_path": repro_md,
        }

    def _handle_validate(self, payload: dict) -> dict:
        from spectraqc.cli import main as cli_main
        from spectraqc.types import Status

        audio_path = _strip_or_none(payload.get("audio_path"))
        profile_path = _strip_or_none(payload.get("profile_path"))
        if not audio_path or not profile_path:
            raise ValueError("audio_path and profile_path are required.")
        fail_on = _strip_or_none(payload.get("fail_on")) or "fail"
        _, decision, profile, _ = cli_main._analyze_audio(
            audio_path,
            profile_path,
            mode="compliance",
        )

        status = decision.overall_status
        summary_lines = [
            f"Profile: {profile.name} ({profile.kind})",
            f"Status: {status.value.upper()}",
        ]
        for bd in decision.band_decisions:
            worst = max(
                bd.mean.status,
                bd.max.status,
                bd.variance.status,
                key=lambda s: {"pass": 0, "warn": 1, "fail": 2}[s.value],
            )
            if worst != Status.PASS:
                summary_lines.append(
                    f"{bd.band.name}: {worst.value} (mean={bd.mean.value:+.2f}dB, max={bd.max.value:.2f}dB)"
                )
        for gd in decision.global_decisions:
            if gd.status != Status.PASS:
                summary_lines.append(f"{gd.metric}: {gd.status.value} ({gd.value:.3f} {gd.units})")

        if fail_on == "warn" and status in (Status.WARN, Status.FAIL):
            exit_code = cli_main._exit_code_for_status(status)
        elif fail_on == "fail" and status == Status.FAIL:
            exit_code = cli_main.EXIT_FAIL
        elif status == Status.FAIL:
            exit_code = cli_main.EXIT_FAIL
        else:
            exit_code = cli_main._exit_code_for_status(status)

        return {
            "status": status.value,
            "exit_code": exit_code,
            "summary_lines": summary_lines,
        }

    def _handle_repair(self, payload: dict) -> dict:
        from spectraqc.cli import main as cli_main
        from spectraqc.dsp.repair import apply_repair_plan, compute_repair_metrics
        from spectraqc.io.audio import load_audio
        from spectraqc.profiles.loader import load_reference_profile
        from spectraqc.utils.hashing import sha256_hex_file
        import numpy as np
        import soundfile as sf

        audio_path = _strip_or_none(payload.get("audio_path"))
        profile_path = _strip_or_none(payload.get("profile_path"))
        repair_plan = _strip_or_none(payload.get("repair_plan"))
        plan_json = _strip_or_none(payload.get("plan_json"))
        if not audio_path or not profile_path:
            raise ValueError("audio_path and profile_path are required.")
        if not repair_plan and not plan_json:
            raise ValueError("repair_plan path or plan_json is required.")
        out_path = _strip_or_none(payload.get("out_path"))
        report_path = _strip_or_none(payload.get("report_path"))

        profile = load_reference_profile(profile_path)
        if plan_json:
            try:
                plan = json.loads(plan_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid plan_json: {exc}") from exc
        else:
            plan = cli_main._load_repair_plan(repair_plan)
        if not isinstance(plan, dict) or "steps" not in plan:
            raise ValueError("repair plan must be a mapping with a 'steps' list.")
        audio = load_audio(audio_path)
        repaired_samples, steps = apply_repair_plan(audio.samples, audio.fs, plan)
        resolved_out_path = Path(out_path) if out_path else Path(audio_path).with_suffix(".repaired.wav")
        resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(resolved_out_path, repaired_samples, int(audio.fs))

        before_metrics = compute_repair_metrics(audio.samples, audio.fs, profile)
        after_metrics = compute_repair_metrics(repaired_samples, audio.fs, profile)
        delta_curve = (
            np.array(after_metrics["deviation_curve_db"], dtype=np.float64)
            - np.array(before_metrics["deviation_curve_db"], dtype=np.float64)
        ).tolist()
        delta_metrics = {
            "true_peak_dbtp": after_metrics["true_peak_dbtp"] - before_metrics["true_peak_dbtp"],
            "noise_floor_dbfs": after_metrics["noise_floor_dbfs"] - before_metrics["noise_floor_dbfs"],
            "deviation_curve_db": delta_curve,
        }
        repair_section = {
            "plan_source": {"path": str(repair_plan)},
            "steps": steps,
            "metrics": {
                "before": before_metrics,
                "after": after_metrics,
                "delta": delta_metrics,
            },
            "output": {
                "path": str(resolved_out_path),
                "file_hash_sha256": sha256_hex_file(str(resolved_out_path)),
                "channels": audio.channels,
                "sample_rate_hz": audio.fs,
            },
        }
        qcreport, decision, _, _ = cli_main._analyze_audio(
            str(resolved_out_path),
            profile_path,
            mode="compliance",
            repair=repair_section,
        )
        output_json = json.dumps(qcreport, indent=2)
        if report_path:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(output_json, encoding="utf-8")

        return {
            "status": decision.overall_status.value,
            "report": qcreport,
            "repaired_audio_path": str(resolved_out_path),
            "report_path": report_path,
        }

    def _handle_repair_plan(self, payload: dict) -> dict:
        from spectraqc.dsp.repair import suggest_repair_plan
        from spectraqc.io.audio import load_audio
        from spectraqc.profiles.loader import load_reference_profile

        audio_path = _strip_or_none(payload.get("audio_path"))
        profile_path = _strip_or_none(payload.get("profile_path"))
        if not audio_path or not profile_path:
            raise ValueError("audio_path and profile_path are required.")

        profile = load_reference_profile(profile_path)
        audio = load_audio(audio_path)
        plan, summary = suggest_repair_plan(audio.samples, audio.fs, profile)

        return {
            "plan": plan,
            "summary": summary,
        }

    def _handle_upload_json(self, payload: dict) -> dict:
        filename = _strip_or_none(payload.get("filename")) or "upload.json"
        content = payload.get("content")
        if content is None:
            raise ValueError("content is required.")
        if not isinstance(content, str):
            raise ValueError("content must be a string.")
        safe_name = Path(filename).name
        if not safe_name.endswith(".json"):
            safe_name = f"{safe_name}.json"
        target = _uploads_dir() / f"{uuid4().hex}_{safe_name}"
        target.write_text(content, encoding="utf-8")
        return {"path": str(target)}

    def _handle_inspect(self, payload: dict) -> dict:
        from spectraqc.profiles.loader import load_reference_profile

        profile_path = _strip_or_none(payload.get("profile_path"))
        if not profile_path:
            raise ValueError("profile_path is required.")
        profile = load_reference_profile(profile_path)

        return {
            "profile": {
                "name": profile.name,
                "kind": profile.kind,
                "version": profile.version,
                "hash": profile.profile_hash_sha256[:16] + "...",
                "band_count": len(profile.bands),
                "grid_bins": len(profile.freqs_hz),
                "freq_min": float(profile.freqs_hz[0]),
                "freq_max": float(profile.freqs_hz[-1]),
                "band_mean_pass": float(profile.thresholds["band_mean_db"]["default"][0]),
                "band_mean_warn": float(profile.thresholds["band_mean_db"]["default"][1]),
                "band_max_pass": float(profile.thresholds["band_max_db"]["all"][0]),
                "band_max_warn": float(profile.thresholds["band_max_db"]["all"][1]),
                "tilt_pass": float(profile.thresholds["tilt_db_per_oct"][0]),
                "tilt_warn": float(profile.thresholds["tilt_db_per_oct"][1]),
            }
        }

    def _handle_batch(self, payload: dict) -> dict:
        from spectraqc.cli import main as cli_main
        from spectraqc.corpus.manifest import load_corpus_manifest
        from spectraqc.reporting.batch_summary import build_batch_summary, build_kpi_payload, render_kpis_csv

        folder = _strip_or_none(payload.get("folder"))
        manifest = _strip_or_none(payload.get("manifest"))
        profile_path = _strip_or_none(payload.get("profile_path"))
        if not profile_path:
            raise ValueError("profile_path is required.")
        mode = _strip_or_none(payload.get("mode")) or "compliance"
        out_dir = _strip_or_none(payload.get("out_dir"))
        if not out_dir:
            raise ValueError("out_dir is required for batch outputs.")
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        audio_paths: list[Path] = []
        if folder:
            audio_paths.extend(cli_main._iter_audio_files(Path(folder), bool(payload.get("recursive"))))
        if manifest:
            _, entries, _ = load_corpus_manifest(manifest)
            for entry in entries:
                if entry.exclude:
                    continue
                audio_paths.append(Path(entry.path))
        if not audio_paths:
            raise ValueError("No input files found.")

        max_workers = max(1, _as_int(payload.get("workers"), os.cpu_count() or 2))
        max_workers = min(max_workers, len(audio_paths))

        results: list[tuple[str, str, str | None, dict | None]] = []
        failures = 0
        if max_workers == 1:
            for path in audio_paths:
                result = cli_main._batch_worker((str(path), profile_path, mode, str(out_dir_path)))
                results.append(result)
                if result[2]:
                    failures += 1
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(cli_main._batch_worker, (str(p), profile_path, mode, str(out_dir_path)))
                    for p in audio_paths
                ]
                for fut in as_completed(futures):
                    audio_path, status, err, report = fut.result()
                    results.append((audio_path, status, err, report))
                    if err:
                        failures += 1

        summary = build_batch_summary(results)
        status_counts = summary.get("totals", {}).get("status_counts", {})
        summary_status = "pass"
        if status_counts.get("fail", 0) > 0:
            summary_status = "fail"
        elif status_counts.get("warn", 0) > 0:
            summary_status = "warn"
        elif status_counts.get("error", 0) > 0:
            summary_status = "error"

        outputs: dict[str, str] = {}
        summary_json_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("summary_json")),
            "batch-summary.json",
        )
        if summary_json_path:
            summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            outputs["summary_json"] = str(summary_json_path)
        summary_md_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("summary_md")),
            "batch-summary.md",
        )
        if summary_md_path:
            summary_md_path.write_text(cli_main._render_markdown_summary(summary), encoding="utf-8")
            outputs["summary_md"] = str(summary_md_path)
        kpis_json_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("summary_kpis_json")),
            "batch-kpis.json",
        )
        if kpis_json_path:
            kpis_payload = build_kpi_payload(summary)
            kpis_json_path.write_text(json.dumps(kpis_payload, indent=2), encoding="utf-8")
            outputs["summary_kpis_json"] = str(kpis_json_path)
        kpis_csv_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("summary_kpis_csv")),
            "batch-kpis.csv",
        )
        if kpis_csv_path:
            kpis_csv_path.write_text(render_kpis_csv(summary), encoding="utf-8")
            outputs["summary_kpis_csv"] = str(kpis_csv_path)
        report_md_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("report_md")),
            "batch-report.md",
        )
        if report_md_path:
            report_md_path.write_text(
                cli_main._render_corpus_report_md(summary, profile_path=profile_path, mode=mode),
                encoding="utf-8",
            )
            outputs["report_md"] = str(report_md_path)
        report_html_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("report_html")),
            "batch-report.html",
        )
        if report_html_path:
            file_links = []
            embedded_reports = {}
            for audio_path, status, _, report in results:
                label = Path(audio_path).name
                link_path = cli_main._output_path(out_dir_path, Path(audio_path))
                href = os.path.relpath(link_path, report_html_path.parent)
                file_links.append((label, status, href))
                if report:
                    embedded_reports[label] = report
            report_html_path.write_text(
                cli_main._render_corpus_report_html(
                    summary,
                    profile_path=profile_path,
                    mode=mode,
                    file_links=file_links,
                    embedded_reports=embedded_reports,
                ),
                encoding="utf-8",
            )
            outputs["report_html"] = str(report_html_path)
        repro_md_path = cli_main._resolve_report_output_path(
            out_dir_path,
            _strip_keep_empty(payload.get("repro_md")),
            "batch-repro.md",
        )
        if repro_md_path:
            profile = cli_main.load_reference_profile(profile_path)
            algo_registry = cli_main.build_algorithm_registry(
                analysis_lock=profile.analysis_lock or {},
                smoothing_cfg=profile.thresholds.get("_smoothing", {"type": "none"}),
                channel_policy=str(profile.analysis_lock.get("channel_policy", "mono")),
            )
            algo_ids = cli_main.algorithm_ids_from_registry(algo_registry)
            repro_md_path.write_text(
                cli_main._render_repro_doc(
                    profile=profile,
                    algorithm_ids=algo_ids,
                    profile_path=profile_path,
                    mode=mode,
                    manifest_path=manifest,
                    folder_path=folder,
                    recursive=bool(payload.get("recursive")),
                ),
                encoding="utf-8",
            )
            outputs["repro_md"] = str(repro_md_path)

        return {
            "summary": summary,
            "status": summary_status,
            "total_files": len(audio_paths),
            "outputs": outputs,
            "failures": failures,
        }

    def _handle_build_profile(self, payload: dict) -> dict:
        from spectraqc.profiles.builder import (
            build_reference_profile_from_folder,
            build_reference_profile_from_manifest,
        )

        manifest_path = _strip_or_none(payload.get("manifest_path"))
        folder_path = _strip_or_none(payload.get("folder_path"))
        if bool(manifest_path) == bool(folder_path):
            raise ValueError("Provide exactly one of manifest_path or folder_path.")
        profile_name = _strip_or_none(payload.get("profile_name")) or "streaming_generic_v1"
        profile_kind = _strip_or_none(payload.get("profile_kind")) or "streaming"
        out_path = _strip_or_none(payload.get("out_path"))
        recursive = bool(payload.get("recursive"))

        if manifest_path:
            profile, output_path = build_reference_profile_from_manifest(
                manifest_path,
                profile_name=profile_name,
                profile_kind=profile_kind,
                output_path=out_path,
            )
            source = {"manifest_path": manifest_path}
        else:
            profile, output_path = build_reference_profile_from_folder(
                folder_path,
                recursive=recursive,
                profile_name=profile_name,
                profile_kind=profile_kind,
                output_path=out_path,
            )
            source = {"folder_path": folder_path, "recursive": recursive}

        return {
            "profile_path": str(output_path),
            "profile_name": profile["profile"]["name"],
            "profile_kind": profile["profile"]["kind"],
            "profile_version": profile["profile"]["version"],
            "profile_hash": profile["integrity"]["profile_hash_sha256"],
            "band_count": len(profile["bands"]),
            "grid_bins": len(profile["frequency_grid"]["freqs_hz"]),
            "file_count": profile["corpus_stats"]["file_count"],
            "source": source,
        }


def run_gui_server(*, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), SpectraQCGUIHandler)
    url = f"http://{host}:{port}/"
    print(f"SpectraQC GUI running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
