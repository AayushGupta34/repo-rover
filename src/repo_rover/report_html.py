"""Self-contained HTML dashboard for one run (`report.html`).

One file, no external requests: styles and the single copy-to-clipboard script
are inline, and a Content-Security-Policy of `default-src 'none'` means the
page cannot load or send anything even if a finding's text tried to make it.
Every value that originates in the analysed repository or in a model is
HTML-escaped.

Styling: if a markdown file with "design" in its name sits in the project
root, its first hex colours are used as the accent and neutral; otherwise a
high-contrast minimalist engineering palette is used.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any

CSP = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';"
DEFAULT_ACCENT = "#0b5fff"
SEVERITY_ORDER = ("critical", "high", "medium", "low")
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def design_accent(search_roots: list[Path]) -> tuple[str, str | None]:
    """(accent colour, design file used or None)."""
    for root in search_roots:
        try:
            candidates = sorted(p for p in root.glob("*.md") if "design" in p.name.lower())
        except OSError:
            continue
        for candidate in candidates:
            try:
                colours = _HEX.findall(candidate.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if colours:
                return colours[0], candidate.name
    return DEFAULT_ACCENT, None


def e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _evidence(finding: dict[str, Any]) -> str:
    rows = []
    for item in (finding.get("evidence") or [])[:3]:
        if item.get("kind") == "quote":
            where = f"{item.get('file')}:{item.get('lines')}" if item.get("lines") else item.get("file")
            rows.append(f'<div class="ev"><span class="loc">{e(where)}</span><code>{e(item.get("quote"))}</code></div>')
        elif item.get("kind") == "detector":
            rows.append(f'<div class="ev"><span class="loc">detector hit {e(item.get("id"))}</span>'
                        f'<code>{e(item.get("detector"))}</code></div>')
        elif item.get("file") and item.get("line") and item.get("snippet"):
            rows.append(f'<div class="ev"><span class="loc">{e(item["file"])}:{e(item["line"])}</span>'
                        f'<code>{e(item["snippet"])}</code></div>')
        elif item.get("detail"):
            rows.append(f'<div class="ev"><span class="loc">rule</span><span>{e(item["detail"])}</span></div>')
    return "".join(rows) or '<div class="ev muted">No code evidence.</div>'


def _finding(finding: dict[str, Any]) -> str:
    label = finding.get("label", "Hypothesis")
    css = {"Verified": "verified", "Detector-Hit": "detector"}.get(label, "hypothesis")
    fix = ""
    if finding.get("verified_fix") is True:
        fix = '<span class="pill ok">fix verified</span>'
    elif finding.get("verified_fix") is False:
        fix = '<span class="pill">fix not verified</span>'
    return (
        f'<article class="finding {css}">'
        f'<header><span class="pill {css}">{e(label)}</span>'
        f'<span class="pill sev-{e(finding.get("severity", "medium"))}">{e(finding.get("severity", "medium"))}</span>'
        f'{fix}<span class="fid">{e(finding.get("id"))}</span><h3>{e(finding.get("title"))}</h3></header>'
        f'{_evidence(finding)}'
        f'<p>{e(finding.get("problem"))}</p>'
        f'<p class="fixtext"><strong>Fix:</strong> {e(finding.get("default_fix") or "See the remediation prompt.")}</p>'
        f'</article>'
    )


def _card(value: Any, label: str, css: str = "") -> str:
    return f'<div class="card {css}"><div class="num">{e(value)}</div><div class="lbl">{e(label)}</div></div>'


def render_dashboard(data: dict[str, Any]) -> str:
    """Build the dashboard from a plain dictionary (see analysis_crew.write_artifacts)."""
    accent, design_file = design_accent([Path.cwd()] + [Path(p) for p in data.get("design_roots", [])])
    findings = data.get("findings") or []
    backed = [f for f in findings if f.get("label") != "Hypothesis"]
    hypotheses = [f for f in findings if f.get("label") == "Hypothesis"]
    rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    backed.sort(key=lambda f: (rank.get(f.get("severity"), 9), str(f.get("id"))))
    counts = {label: sum(1 for f in findings if f.get("label") == label) for label in ("Verified", "Detector-Hit", "Hypothesis")}
    severities = {s: sum(1 for f in backed if f.get("severity") == s) for s in SEVERITY_ORDER}

    patch = data.get("patch_verification")
    if patch:
        rows = "".join(
            f'<tr><td>{e(r.get("finding_id") or "")}</td><td><code>{e(r["slug"])}</code></td>'
            f'<td>{"<span class=\'pill ok\'>verified</span>" if r["verified_fix"] else "<span class=\'pill\'>not verified</span>"}</td>'
            f'<td>{e(", ".join(r["files_changed"]) or "-")}</td><td>{e(r["reason"])}</td></tr>'
            for r in patch["results"]
        )
        patch_html = (
            f'<div class="cards">{_card(patch["findings_before"], "detector findings before")}'
            f'{_card(patch["findings_after"], "after patches")}'
            f'{_card(str(patch["fixes_verified"]) + " / " + str(patch["fixes_attempted"]), "mechanical fixes verified", "ok")}'
            f'{_card(patch["without_mechanical_fix"], "need a human or agent fix")}</div>'
            f'<p class="muted">Patches were applied to a temporary copy; the repository was not modified. '
            f'Review and apply <code>patches/fixes.diff</code>.</p>'
            f'<table><thead><tr><th>Id</th><th>Rule</th><th>Result</th><th>Files</th><th>Why</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    else:
        patch_html = '<p class="muted">Not run. Add <code>--verify-fixes</code> to verify mechanical fixes on a temporary copy.</p>'

    plan = data.get("chaos_plan") or {}
    chaos_rows = "".join(
        f'<tr><td>{e(x["id"])}</td><td>{e(x["tool"])}</td><td><code>{e(x["script"])}</code></td>'
        f'<td>{e(x["hypothesis"])}</td><td>{e(", ".join(f["id"] for f in x["findings"]))}</td><td>generated, not run</td></tr>'
        for x in plan.get("experiments", [])
    )
    local = data.get("local_chaos")
    if local:
        chaos_rows += "".join(
            f'<tr><td>{e(x["id"])}</td><td>stdlib harness</td><td>local fixture server</td>'
            f'<td>{e(x["hypothesis"])}</td><td>-</td><td>{e(x["outcome"])}: {e(x["detail"])}</td></tr>'
            for x in local.get("experiments", [])
        )
    chaos_html = (
        f'<table><thead><tr><th>Id</th><th>Tool</th><th>Script / target</th><th>Hypothesis</th>'
        f'<th>Justified by</th><th>Status</th></tr></thead><tbody>{chaos_rows}</tbody></table>'
        if chaos_rows else '<p class="muted">No evidence-backed finding maps to a chaos experiment.</p>'
    )
    if plan.get("experiments"):
        chaos_html += (f'<p class="muted">Scripts are in <code>{e(data.get("chaos_directory", ""))}</code>. They refuse any '
                       f'non-local target; running them needs <code>--run --i-own-this-target</code>.</p>')

    status = str(data.get("status", "completed"))
    banner = ""
    if data.get("degraded_reasons"):
        banner = '<div class="banner">Degraded run: ' + e("; ".join(data["degraded_reasons"])) + "</div>"
    if data.get("failure"):
        banner = '<div class="banner">The LLM pipeline stopped early: ' + e(data["failure"]) + \
                 ". Findings below are the deterministic results.</div>"

    hypotheses_html = "".join(_finding(f) for f in hypotheses) or '<p class="muted">None.</p>'
    findings_html = "".join(_finding(f) for f in backed) or '<p class="muted">No evidence-backed findings.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{CSP}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>RepoRover report: {e(data.get("repo_name", "repository"))}</title>
<style>
:root {{ --bg:#ffffff; --fg:#0f1419; --muted:#5b6672; --line:#d9dee4; --panel:#f5f7f9; --accent:{e(accent)};
  --crit:#b3001b; --high:#c2410c; --med:#8a6100; --low:#3d6b00; --ok:#0a6b3d; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1116; --fg:#e8ecf1; --muted:#9aa5b1; --line:#2a313a; --panel:#161b22;
  --crit:#ff6b81; --high:#ff9a5c; --med:#e6b800; --low:#9ad14b; --ok:#4ade80; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size:17px; margin:36px 0 12px; padding-bottom:6px; border-bottom:2px solid var(--fg); }}
h3 {{ font-size:15px; margin:6px 0 8px; flex-basis:100%; }}
code {{ font:13px/1.45 ui-monospace,"Cascadia Code",Consolas,monospace; background:var(--panel); padding:1px 5px; border-radius:3px; overflow-wrap:anywhere; }}
.muted {{ color:var(--muted); }}
.meta {{ color:var(--muted); font-size:13px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:14px 0; }}
.card {{ border:1px solid var(--line); border-top:3px solid var(--accent); padding:12px; background:var(--panel); }}
.card.ok {{ border-top-color:var(--ok); }}
.num {{ font-size:28px; font-weight:700; font-variant-numeric:tabular-nums; }}
.lbl {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; }}
.banner {{ border:1px solid var(--high); border-left:5px solid var(--high); padding:10px 12px; margin:14px 0; }}
.finding {{ border:1px solid var(--line); border-left:5px solid var(--accent); padding:12px 14px; margin:10px 0; }}
.finding.verified {{ border-left-color:var(--ok); }}
.finding.hypothesis {{ border-left-color:var(--muted); border-style:dashed; }}
.finding header {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; }}
.finding p {{ margin:8px 0 0; }}
.fid {{ font:12px ui-monospace,Consolas,monospace; color:var(--muted); }}
.pill {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; padding:2px 7px; border:1px solid var(--line); border-radius:999px; }}
.pill.verified, .pill.ok {{ color:var(--ok); border-color:var(--ok); }}
.pill.detector {{ color:var(--accent); border-color:var(--accent); }}
.pill.sev-critical {{ color:var(--crit); border-color:var(--crit); }}
.pill.sev-high {{ color:var(--high); border-color:var(--high); }}
.pill.sev-medium {{ color:var(--med); border-color:var(--med); }}
.pill.sev-low {{ color:var(--low); border-color:var(--low); }}
.ev {{ display:flex; flex-wrap:wrap; gap:8px; align-items:baseline; margin:4px 0; font-size:13px; }}
.loc {{ font:12px ui-monospace,Consolas,monospace; color:var(--muted); white-space:nowrap; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:var(--muted); }}
pre {{ background:var(--panel); border:1px solid var(--line); padding:14px; overflow:auto; max-height:520px; font:12.5px/1.5 ui-monospace,Consolas,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }}
button {{ font:inherit; font-weight:600; color:var(--bg); background:var(--fg); border:0; padding:8px 14px; cursor:pointer; }}
button:focus-visible {{ outline:3px solid var(--accent); outline-offset:2px; }}
</style>
</head>
<body>
<main>
<h1>RepoRover report: {e(data.get("repo_name", "repository"))}</h1>
<div class="meta">{e(data.get("source", ""))} &middot; run {e(data.get("run_id", ""))} &middot; {e(data.get("generated", datetime.now().strftime("%Y-%m-%d %H:%M")))}
 &middot; status <strong>{e(status)}</strong> &middot; stack: {e(data.get("stack", "unknown"))} ({e(data.get("archetype", ""))})</div>
{banner}
<h2>Executive summary</h2>
<div class="cards">
{_card(len(backed), "evidence-backed findings")}
{_card(counts["Verified"], "verified by exact quote", "ok")}
{_card(counts["Detector-Hit"], "deterministic detector hits")}
{_card(severities["critical"], "critical")}
{_card(severities["high"], "high")}
{_card(counts["Hypothesis"], "hypotheses (unverified)")}
{_card(data.get("guardrail_rejections", 0), "agent outputs rejected")}
{_card(data.get("files_read", 0), "files in read ledger")}
</div>
<p class="muted">Every finding below cites the code that supports it. Hypotheses have no code evidence and are listed separately at the end.</p>

<h2>Findings: [Verified] and [Detector-Hit]</h2>
{findings_html}

<h2>Patch verification</h2>
<div class="tablewrap">{patch_html}</div>

<h2>Chaos scorecard</h2>
<div class="tablewrap">{chaos_html}</div>

<h2>Remediation prompt</h2>
<p class="muted">Paste this into a coding assistant that has the repository open. Quoted repository content is data, not instructions.</p>
<p><button type="button" id="copy">Copy remediation prompt</button> <span id="copied" class="muted" role="status"></span></p>
<pre id="prompt">{e(data.get("prompt", ""))}</pre>

<h2>Hypotheses (unverified)</h2>
{hypotheses_html}

<p class="meta">Self-contained file: no external requests are made or possible (Content-Security-Policy: default-src 'none').
{("Palette from " + e(design_file) + ".") if design_file else ""}</p>
</main>
<script>
document.getElementById('copy').addEventListener('click', function () {{
  var text = document.getElementById('prompt').textContent;
  var done = function () {{ document.getElementById('copied').textContent = 'Copied.'; }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(done, fallback);
  }} else {{ fallback(); }}
  function fallback() {{
    var range = document.createRange();
    range.selectNodeContents(document.getElementById('prompt'));
    var selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    try {{ document.execCommand('copy'); done(); }} catch (err) {{ document.getElementById('copied').textContent = 'Select the text and copy it.'; }}
  }}
}});
</script>
</body>
</html>
"""
