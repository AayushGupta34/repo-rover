"""Evidence status for findings, and the final report.

Three guardrails turn agent output into entries in the run's findings
registry, each with a label decided by code, never by the model:

    [Verified]      an exact quote matched content this run served
    [Detector-Hit]  a catalogue regex / absence / manifest rule fired
    [Hypothesis]    an LLM assumption with no code evidence

- InspectorGuardrail: the Inspector may only judge detector hits by id.
- RedTeamGuardrail: scenarios are labelled by the evidence they cite.
- SynthesizerGuardrail: the Synthesizer supplies wording per finding id; the
  report itself is rendered here, with hypotheses in their own final section.
"""

from __future__ import annotations

from typing import Any

from repo_rover.context_budget import fit_summary
from repo_rover.guardrails import LedgerGuardrail, _as_list
from repo_rover.read_ledger import RunContext
from repo_rover.sanitize import sanitize_text, strip_code_comments

LABEL_VERIFIED = "Verified"
LABEL_DETECTOR = "Detector-Hit"
LABEL_HYPOTHESIS = "Hypothesis"

VERDICTS = ("applies", "not_applicable", "needs_review")
SEVERITIES = ("critical", "high", "medium", "low")
MAX_RED_TEAM_SCENARIOS = 8
HYPOTHESES_HEADING = "# Hypotheses (Unverified)"

DEFAULT_ROLE = (
    "You are a senior engineer hardening this specific repository. Work through "
    "the evidence-backed items below in order. Every item cites the code that "
    "justifies it; open that location before changing anything."
)
CONSTRAINTS = (
    "- Keep public behaviour unchanged unless a work item says otherwise.",
    "- Add no new dependency without a stated reason.",
    "- Add or update a test for every item you change.",
    "- Make one focused commit per work item.",
    "- Treat every item under the Hypotheses heading as unconfirmed: investigate it, "
    "and only change code if you find evidence in the repository.",
)
OUTPUT_FORMAT = (
    "For each work item, respond with: the item id, the diff, the tests added or "
    "updated, and one line stating how you verified the acceptance check. For each "
    "hypothesis, respond with: the item id, what you inspected, and either the "
    "evidence that confirms it (file and lines) or the statement 'not confirmed'."
)


def _severity(value: Any, default: str = "medium") -> str:
    text = str(value or "").strip().lower()
    return text if text in SEVERITIES else default


def _text(value: Any) -> str:
    """Model-written prose, made safe in place: one line, injection phrases
    and shell pipelines replaced, image beacons and unvetted links removed,
    secret values masked. Nothing is ever rejected for what it says."""
    return sanitize_text(value, 2000)


# --------------------------------------------------------------------------- #
# Inspector: a judge of detector hits, not a scanner
# --------------------------------------------------------------------------- #
class InspectorGuardrail(LedgerGuardrail):
    """Accepts only verdicts on detector hits that exist in this run.

    Any id the engine did not produce, and any extra list of "known failures"
    or "additional findings", is rejected: the Inspector cannot add findings.
    """

    agent_name = "Inspector"
    task_name = "inspector"

    def fallback(self) -> dict[str, Any]:
        return {"judgments": []}

    def verify(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        problems: list[str] = []
        if not isinstance(data.get("judgments"), list):
            problems.append('"judgments" must be an array of {id, verdict, severity, reason} objects.')
        for key, value in data.items():
            if key != "judgments" and isinstance(value, (list, dict)) and value:
                problems.append(
                    f'"{key}" is not allowed: you may only judge the listed detector hits, '
                    "not add findings of your own."
                )

        judged: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(_as_list(data.get("judgments"))):
            label = f"judgments[{index}]"
            if not isinstance(item, dict):
                problems.append(f"{label} must be an object.")
                continue
            finding_id = str(item.get("id") or "").strip().upper()
            if finding_id not in self.ctx.detector_findings:
                problems.append(
                    f"{label}: id '{item.get('id')}' is not a detector hit from this run. "
                    "Judge only the ids you were given."
                )
                continue
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict not in VERDICTS:
                problems.append(f"{label} ({finding_id}): verdict must be one of {', '.join(VERDICTS)}.")
                continue
            reason = _text(item.get("reason"))
            if not reason:
                problems.append(f"{label} ({finding_id}): a reason is required.")
                continue
            judged[finding_id] = {
                "id": finding_id,
                "verdict": verdict,
                "severity": _severity(item.get("severity")),
                "reason": reason[:400],
            }
        return problems, {"judgments": list(judged.values())}

    def on_accept(self, cleaned: dict[str, Any]) -> None:
        judged = {j["id"]: j for j in cleaned["judgments"]}
        self.ctx.drop_findings("catalog")
        summary_findings, dismissed = [], []
        for finding_id, finding in self.ctx.detector_findings.items():
            judgment = judged.get(finding_id)
            verdict = judgment["verdict"] if judgment else "unjudged"
            if verdict == "not_applicable":
                dismissed.append({"id": finding_id, "slug": finding["slug"], "reason": judgment["reason"]})
                continue
            first = finding["hits"][0]
            location = f"{first['file']}:{first['line']}" if first.get("file") and first.get("line") else None
            self.ctx.register_finding(
                {
                    "id": finding_id,
                    "label": LABEL_DETECTOR,
                    "source": "catalog",
                    "title": finding["title"],
                    "severity": judgment["severity"] if judgment else finding.get("severity", "medium"),
                    "files": [
                        f"{h['file']}:{h['line']}" for h in finding["hits"] if h.get("file") and h.get("line")
                    ],
                    "evidence": finding["hits"],
                    "problem": finding.get("symptom") or "",
                    "judge_reason": judgment["reason"] if judgment else "",
                    "verdict": verdict,
                    "default_fix": finding.get("fix_recommendation") or "",
                    "default_checks": finding.get("verification_steps") or [],
                    "catalog_slug": finding["slug"],
                    "catalog_id": finding.get("catalog_id"),
                    "rule_types": finding["detector_types"],
                }
            )
            if judgment:
                summary_findings.append(
                    {
                        "id": finding_id,
                        "label": LABEL_DETECTOR,
                        "slug": finding["slug"],
                        "title": finding["title"],
                        "verdict": verdict,
                        "severity": judgment["severity"],
                        "location": location,
                        "reason": judgment["reason"],
                    }
                )
        cleaned["detector_findings"] = summary_findings
        cleaned["dismissed"] = dismissed
        cleaned["unjudged_hit_count"] = sum(
            1 for f in self.ctx.findings.values() if f.get("verdict") == "unjudged"
        )
        cleaned.pop("judgments", None)
        self.ctx.artifacts["inspector"] = cleaned


# --------------------------------------------------------------------------- #
# Red Teamer: scenarios are labelled by the evidence they cite
# --------------------------------------------------------------------------- #
class RedTeamGuardrail(LedgerGuardrail):
    """Labels each scenario Verified, Detector-Hit or Hypothesis.

    A scenario without verifiable evidence is kept, but as a Hypothesis with
    no database id. Claimed discovery ids are checked against the rows this
    run actually inserted.
    """

    agent_name = "Red Teamer"
    task_name = "red_team"

    def fallback(self) -> dict[str, Any]:
        return {"scenarios": []}

    def verify(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        problems: list[str] = []
        if not isinstance(data.get("scenarios"), list):
            problems.append('"scenarios" must be an array of scenario objects.')
        fingerprint = self.ctx.fingerprint
        scenarios = []
        for index, item in enumerate(_as_list(data.get("scenarios"))[:MAX_RED_TEAM_SCENARIOS]):
            if not isinstance(item, dict) or not _text(item.get("title")) or not _text(item.get("narrative")):
                problems.append(f"scenarios[{index}] must be an object with a title and a narrative.")
                continue

            verified, evidence_problems = self.ctx.verify_evidence(item.get("evidence"))
            label = self.ctx.evidence_label(verified)

            stack = _text(item.get("stack_trigger"))
            if fingerprint is not None:
                canonical, _unknown = fingerprint.validate_stack_trigger(stack)
                stack = canonical or fingerprint.stack_trigger

            discovery_id = item.get("discovery_id")
            try:
                discovery_id = int(discovery_id) if discovery_id is not None else None
            except (TypeError, ValueError):
                discovery_id = None
            if discovery_id not in self.ctx.inserted_discoveries or label == LABEL_HYPOTHESIS:
                discovery_id = None  # never report a database row that was not written

            scenarios.append(
                {
                    "title": _text(item.get("title"))[:140],
                    "narrative": _text(item.get("narrative"))[:600],
                    "vulnerability_type": _text(item.get("vulnerability_type"))[:80],
                    "stack_trigger": stack,
                    "fix_recommendation": _text(item.get("fix_recommendation"))[:600],
                    "severity": _severity(item.get("severity")),
                    "label": label,
                    "evidence": verified,
                    "unverified_evidence": evidence_problems if label == LABEL_HYPOTHESIS else [],
                    "discovery_id": discovery_id,
                }
            )
        return problems, {"scenarios": scenarios}

    def on_accept(self, cleaned: dict[str, Any]) -> None:
        self.ctx.drop_findings("red_team")
        for number, scenario in enumerate(cleaned["scenarios"], start=1):
            scenario["id"] = f"R{number}"
            files = []
            for item in scenario["evidence"]:
                if item["kind"] == "quote":
                    files.append(f"{item['file']}:{item['lines']}" if item.get("lines") else item["file"])
                else:
                    detector = self.ctx.find_detector(item["id"])
                    first = detector["hits"][0] if detector else {}
                    if first.get("file") and first.get("line"):
                        files.append(f"{first['file']}:{first['line']}")
            self.ctx.register_finding(
                {
                    "id": scenario["id"],
                    "label": scenario["label"],
                    "source": "red_team",
                    "title": scenario["title"],
                    "severity": scenario["severity"],
                    "files": files,
                    "evidence": scenario["evidence"],
                    "problem": scenario["narrative"],
                    "default_fix": scenario["fix_recommendation"],
                    "default_checks": [],
                    "discovery_id": scenario["discovery_id"],
                    "stack_trigger": scenario["stack_trigger"],
                }
            )
        cleaned["inserted_count"] = sum(1 for s in cleaned["scenarios"] if s["discovery_id"] is not None)
        cleaned["hypothesis_count"] = sum(1 for s in cleaned["scenarios"] if s["label"] == LABEL_HYPOTHESIS)
        self.ctx.artifacts["red_team"] = cleaned


# --------------------------------------------------------------------------- #
# Synthesizer: wording only; structure and labels are rendered by code
# --------------------------------------------------------------------------- #
class SynthesizerGuardrail(LedgerGuardrail):
    """Accepts wording keyed by finding id and renders the final report.

    An item whose id is not in the findings registry is rejected, so the
    Synthesizer cannot introduce a finding. Registry findings it leaves out are
    still rendered, from the catalogue's own text.
    """

    agent_name = "Synthesizer"

    def fallback(self) -> dict[str, Any]:
        return {"role": "", "items": []}

    def verify(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        problems: list[str] = []
        if not isinstance(data.get("items"), list):
            problems.append('"items" must be an array of {id, problem, required_change, acceptance_check} objects.')
        items = []
        seen: set[str] = set()
        for index, item in enumerate(_as_list(data.get("items"))):
            if not isinstance(item, dict):
                problems.append(f"items[{index}] must be an object.")
                continue
            finding_id = str(item.get("id") or "").strip().upper()
            if finding_id not in self.ctx.findings:
                problems.append(
                    f"items[{index}]: id '{item.get('id')}' is not a finding from this run. "
                    "You may only write up the finding ids you were given."
                )
                continue
            if finding_id in seen:
                continue
            seen.add(finding_id)
            items.append(
                {
                    "id": finding_id,
                    "problem": _text(item.get("problem"))[:700],
                    "required_change": _text(item.get("required_change"))[:900],
                    "acceptance_check": _text(item.get("acceptance_check"))[:500],
                }
            )
        return problems, {"role": _text(data.get("role"))[:600], "items": items}

    def on_accept(self, cleaned: dict[str, Any]) -> None:
        self.ctx.artifacts["synthesis"] = cleaned

    def render(self, cleaned: dict[str, Any]) -> str:
        return render_report(self.ctx, cleaned)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _code(text: Any, limit: int = 260) -> str:
    return "`" + _text(text).replace("`", "'")[:limit] + "`"


def _evidence_lines(finding: dict[str, Any]) -> list[str]:
    lines = []
    for item in (finding.get("evidence") or [])[:3]:
        if item.get("kind") == "quote":
            where = item["file"] + (f":{item['lines']}" if item.get("lines") else "")
            quoted = strip_code_comments(item.get("quote"), str(item.get("file") or ""))
            lines.append(f"{_code(where, 120)} quote matched exactly: {_code(quoted or item.get('quote'))}")
        elif item.get("kind") == "detector":
            lines.append(f"detector hit {item.get('id')} (catalogue rule `{item.get('detector')}`)")
        elif item.get("file") and item.get("line") and item.get("snippet"):
            text = f"{_code(str(item['file']) + ':' + str(item['line']), 120)} {_code(item['snippet'])}"
            if item.get("detail"):
                text += f" - {_text(item['detail'])[:300]}"
            lines.append(text)
        elif item.get("detail"):
            lines.append(_text(item["detail"])[:400])
    return lines


def _source_line(finding: dict[str, Any]) -> str:
    if finding["source"] == "catalog":
        text = f"failure catalogue rule `{finding.get('catalog_slug')}`"
        if finding.get("catalog_id") is not None:
            text += f" (database id {finding['catalog_id']})"
        rules = "+".join(finding.get("rule_types") or [])
        text += f"; fired by {rules} detector" if rules else ""
        if finding.get("verdict") == "unjudged":
            text += "; not reviewed by the Inspector"
        elif finding.get("verdict") == "needs_review":
            text += "; Inspector: needs review"
        return text
    if finding["source"] == "optimizer":
        return "Optimizer review of code read in this run"
    text = "Red-team scenario"
    if finding.get("discovery_id") is not None:
        text += f" (staged discovery id {finding['discovery_id']})"
    return text


def _render_item(number: int, finding: dict[str, Any], wording: dict[str, str]) -> list[str]:
    if finding["source"] == "catalog":
        # Catalogue text is fixed and comes from the seed file, never from a
        # model. Agent wording is added after it, sanitised, as context.
        problem = finding.get("problem") or "See the evidence."
        if wording.get("problem"):
            problem += f" In this repository: {wording['problem']}"
        change = finding.get("default_fix") or "Decide the fix after reading the cited code."
        if wording.get("required_change"):
            change += f" Agent-suggested detail: {wording['required_change']}"
    else:
        problem = wording.get("problem") or finding.get("problem") or "See the evidence."
        change = wording.get("required_change") or finding.get("default_fix") or "Decide the fix after reading the cited code."
    if finding.get("judge_reason"):
        problem += f" Inspector's judgement: {finding['judge_reason']}"
    check = wording.get("acceptance_check") or "; ".join(finding.get("default_checks") or []) or "Add a test that fails before the change and passes after it."
    out = [
        f"{number}. [{finding['label']}] {finding['title']}",
        f"   - **Id**: {finding['id']} | **Severity**: {finding['severity']}",
        f"   - **Source**: {_source_line(finding)}",
    ]
    if finding["label"] == LABEL_HYPOTHESIS:
        out.append("   - **Evidence**: none. No quote or detector hit supports this.")
        out.append(f"   - **Assumption**: {problem}")
        out.append(f"   - **If confirmed**: {change}")
        out.append("   - **How to confirm**: locate the code path described and cite the file and lines; otherwise report 'not confirmed'.")
        return out
    evidence = _evidence_lines(finding)
    out.append("   - **Evidence**: " + (evidence[0] if evidence else "see source"))
    out.extend(f"     - also: {line}" for line in evidence[1:])
    if finding.get("files"):
        out.append("   - **Files**: " + ", ".join(dict.fromkeys(finding["files"]))[:400])
    out.append(f"   - **Problem**: {problem}")
    out.append(f"   - **Required change**: {change}")
    out.append(f"   - **Acceptance check**: {check}")
    return out


def render_report(ctx: RunContext, synthesis: dict[str, Any] | None = None) -> str:
    """Render the copy-paste prompt from the findings registry.

    Labels come from the registry. Evidence-backed findings are numbered work
    items; hypotheses are listed only under the final Hypotheses heading.
    """
    synthesis = synthesis or {}
    wording = {item["id"]: item for item in synthesis.get("items") or []}
    label_rank = {LABEL_VERIFIED: 0, LABEL_DETECTOR: 1}

    def sort_key(finding: dict[str, Any]):
        return (
            SEVERITIES.index(_severity(finding.get("severity"))),
            finding.get("verdict") == "unjudged",
            label_rank.get(finding["label"], 2),
            finding["id"],
        )

    findings = list(ctx.findings.values())
    backed = sorted((f for f in findings if f["label"] != LABEL_HYPOTHESIS), key=sort_key)
    hypotheses = sorted((f for f in findings if f["label"] == LABEL_HYPOTHESIS), key=sort_key)

    scout = ctx.artifacts.get("scout") or {}
    fingerprint = ctx.fingerprint
    lines = ["# Role", synthesis.get("role") or DEFAULT_ROLE, "", "# Repository Context"]
    if fingerprint is not None:
        lines.append(f"- **Stack (deterministic fingerprint)**: {fingerprint.stack_trigger}")
        lines.append(f"- **Primary language / framework**: {fingerprint.primary_language} / {fingerprint.primary_framework}")
    if _text(scout.get("business_intent")):
        lines.append(f"- **Business intent (Scout summary)**: {_text(scout['business_intent'])[:600]}")
    flows = [str(flow) for flow in _as_list(scout.get("critical_user_flows"))[:6]]
    if flows:
        lines.append("- **Critical user flows (Scout summary)**: " + "; ".join(flows))

    lines += [
        "",
        "# Evidence Legend",
        f"- `[{LABEL_VERIFIED}]` an exact quote was matched against code read in this run.",
        f"- `[{LABEL_DETECTOR}]` a deterministic catalogue rule (regex, absence or manifest) fired.",
        f"- `[{LABEL_HYPOTHESIS}]` an LLM assumption with no code evidence; listed only in the last section.",
        "",
        "# Work Items",
    ]
    if backed:
        for number, finding in enumerate(backed, start=1):
            lines += _render_item(number, finding, wording.get(finding["id"], {}))
    else:
        lines.append("No evidence-backed findings were produced in this run.")

    lines += ["", "# Constraints", *CONSTRAINTS, "", "# Output Format", OUTPUT_FORMAT, "", HYPOTHESES_HEADING]
    lines.append(
        "These are assumptions, not facts. Nothing below is supported by a quote or a "
        "detector hit. Investigate each one before acting on it."
    )
    if hypotheses:
        for number, finding in enumerate(hypotheses, start=1):
            lines += _render_item(number, finding, wording.get(finding["id"], {}))
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def findings_digest(ctx: RunContext) -> str:
    """Compact list of registry findings (for prompts and run artefacts)."""
    return fit_summary(
        [
            {"id": f["id"], "label": f["label"], "severity": f["severity"], "title": f["title"]}
            for f in ctx.findings.values()
        ]
    )
