from dotenv import load_dotenv

load_dotenv()

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai_tools import DirectoryReadTool

from repo_rover.chaos.generator import generate_plan
from repo_rover.checkpoint import RunCheckpoint
from repo_rover.context_budget import (
    AGENT_SUMMARY_SHARE,
    FINGERPRINT_SHARE,
    budget_tokens,
    truncate_to_tokens,
)
from repo_rover.db_tools import (
    InsertDiscoveryTool,
    finish_analysis_run,
    start_analysis_run,
)
from repo_rover.detector_engine import (
    format_findings_for_judge,
    load_catalog_entries,
    run_detectors,
)
from repo_rover.evidence_report import (
    InspectorGuardrail,
    RedTeamGuardrail,
    SynthesizerGuardrail,
    render_report,
)
from repo_rover.fingerprint import fingerprint_repo
from repo_rover.guardrails import OptimizerGuardrail, ScoutGuardrail
from repo_rover.read_ledger import RunContext, get_active_run, set_active_run
from repo_rover.repo_tools import build_repo_tools
from repo_rover.runtime import GroqChatLLM, build_chat, build_fallback_chats, require_env

RUNS_DIRECTORY = "reporover_runs"

# The flag vocabulary shared by every agent that reads or writes binary_flags.
BINARY_FLAG_KEYS = (
    "data_loss_risk",
    "security_impact",
    "requires_network",
    "mobile_specific",
    "concurrency_related",
    "user_visible",
    "affects_payments_or_pii",
    "reproducible_offline",
)

FILE_READ_RULES = (
    "File reading rules: pass paths exactly as the directory listing shows them "
    "(relative to the repository root). ReadRepoFileTool returns at most 200 "
    "numbered lines per call - page with start_line for longer files. Use "
    "OutlineFileTool to see a file's structure and SearchRepoTool to locate code "
    "before reading it. The run has a fixed read budget, so read only what you "
    "need. Your answer is checked against a ledger of what these tools returned: "
    "a file you did not read, or a quote that is not copied exactly from the "
    "tool output (without the leading line numbers), is rejected."
)


def run_analyzer(target_directory, resume=False, checkpoint_directory=None):
    """Build the evidence-first analysis crew for a local repository.

    Before any agent is created, two deterministic passes run over the
    repository: the stack fingerprint and the failure catalogue's detectors.
    Their results are placed in the task prompts, so agents judge and explain
    concrete evidence instead of generating findings from scratch.

    Returns the Crew; the caller runs it with `crew.kickoff()`.

    Each task's accepted output is checkpointed under
    `<checkpoint_directory>/.reporover/runs/`. With `resume=True`, a crashed
    run for the same unchanged repository is restored and only the tasks that
    had not finished are put in the crew.

    Side effect: the process working directory is changed to the target
    repository. crewai_tools' DirectoryReadTool only lists paths inside the
    working directory, so the crew must run from inside the repo it analyzes.
    """
    checkpoint_base = Path(checkpoint_directory) if checkpoint_directory else Path.cwd()
    target_path = Path(target_directory).expanduser().resolve()
    if not target_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {target_path}")
    target_directory = str(target_path)

    require_env("LLM_API_KEY")
    require_env("DATABASE_URL")
    os.chdir(target_directory)

    # --- Run state: read ledger, degraded flag, database run record ---------
    run = RunContext(target_directory)
    set_active_run(run)

    # --- Deterministic pre-pass (no LLM) --------------------------------------
    fingerprint = fingerprint_repo(target_directory)
    run.fingerprint = fingerprint
    catalog_entries, catalog_source = load_catalog_entries()
    detector_findings = run_detectors(target_directory, catalog_entries, fingerprint, run)
    hits_block, judged_ids = format_findings_for_judge(detector_findings)
    run.artifacts["catalog_source"] = catalog_source
    run.artifacts["hits_shown_to_inspector"] = judged_ids

    # --- Checkpoint: resume a crashed run, or start a new one ----------------
    checkpoint = RunCheckpoint(checkpoint_base, target_directory)
    completed_outputs: dict[str, str] = {}
    if resume:
        completed_outputs, resume_message = checkpoint.resume_into(run)
        print(f"Checkpoint: {resume_message}")
    if not completed_outputs:
        checkpoint.start(run)
    run.checkpoint = checkpoint
    run.artifacts["resumed_tasks"] = sorted(completed_outputs)

    if run.db_run_id is None:
        run.db_run_id = start_analysis_run(run.run_uuid, target_directory)

    fingerprint_block = truncate_to_tokens(
        fingerprint.summary_text(), budget_tokens(FINGERPRINT_SHARE)
    )
    allowed_stack_terms = ", ".join(sorted(set(fingerprint.allowed_terms().values())))

    # --- LLMs: one OpenAI-compatible chat model per agent (see .env) ---------
    precise_chat = build_chat(0.1)
    creative_chat = build_chat(0.8)
    # Each LLM retries transient failures with backoff and full jitter, then
    # falls back to the smaller models in LLM_FALLBACK_MODELS.
    precise_llm = GroqChatLLM(precise_chat, fallbacks=build_fallback_chats(0.1))
    creative_llm = GroqChatLLM(creative_chat, fallbacks=build_fallback_chats(0.8))

    # --- Tools ---------------------------------------------------------------
    # Every repository tool is confined to target_directory, skips .git,
    # node_modules, virtualenvs, binaries and secret files, enforces hard read
    # limits, and records what it served in the run's read ledger.
    repo_tools = build_repo_tools(run)
    directory_tool: DirectoryReadTool = repo_tools["list"]
    file_tool = repo_tools["read"]
    search_tool = repo_tools["search"]
    outline_tool = repo_tools["outline"]

    scout_guardrail = ScoutGuardrail(run)
    inspector_guardrail = InspectorGuardrail(run)
    red_team_guardrail = RedTeamGuardrail(run)
    optimizer_guardrail = OptimizerGuardrail(run)
    synthesizer_guardrail = SynthesizerGuardrail(run)

    flag_keys = ", ".join(BINARY_FLAG_KEYS)

    # --- Agents --------------------------------------------------------------
    scout = Agent(
        role="Scout - Repository Context Mapper",
        goal=(
            f"Using the deterministic stack fingerprint of {target_directory}, output "
            "a rigid JSON object that defines the real-world business intent of the "
            "codebase, backed by exact quotes."
        ),
        backstory=(
            "You were fired from a health-tech unicorn for reviewing code without "
            "mapping the real-world business logic first - you approved a clean "
            "looking diff that silently broke patient dosage scheduling. You are "
            "now terrified of ever judging code out of context. You strictly map "
            "the stack, the entry points and the business intent before anyone is "
            "allowed to form an opinion, and you only state what the files prove."
        ),
        tools=[directory_tool, outline_tool, search_tool, file_tool],
        llm=precise_llm,
        allow_delegation=False,
        max_iter=18,
        verbose=True,
    )

    inspector = Agent(
        role="Inspector - Known Failure Judge",
        goal=(
            "Judge whether each deterministic detector hit from the failure "
            "catalogue really applies to this repository. Never add a finding "
            "that the detector engine did not report."
        ),
        backstory=(
            "You were fired from Apple for missing an iOS Safari landscape "
            "orientation bug that was already documented in the internal bug "
            "database - you simply never looked it up. You are now paranoid. You "
            "rigorously check the database records for every framework, library "
            "and runtime in the stack before you say a word, and you never invent "
            "a record that the database did not return."
        ),
        tools=[file_tool],
        llm=precise_llm,
        allow_delegation=False,
        max_iter=12,
        verbose=True,
    )

    red_teamer = Agent(
        role="Red Teamer - Extreme Scenario Generator",
        goal=(
            "Propose extreme, use-case-specific failure scenarios that the "
            "catalogue does not cover, each tied to the fingerprinted stack and "
            "to code evidence, and log the evidence-backed ones to the staging "
            "table."
        ),
        backstory=(
            "You were fired from a fintech startup after a cellular-drop race "
            "condition - a user walked into an elevator mid-transfer and the "
            "payment was sent twice. Nobody had imagined it. You are haunted by "
            "'what ifs' and now imagine the worst realistic thing that could "
            "happen to this exact product, with these exact users, on their worst "
            "day."
        ),
        tools=[search_tool, file_tool, InsertDiscoveryTool],
        llm=creative_llm,
        allow_delegation=False,
        max_iter=18,
        verbose=True,
    )

    optimizer = Agent(
        role="Optimizer - Complexity Enforcer",
        goal=(
            "Hunt down bloated code in the repository and enforce strict Big-O "
            "limits on every hot path."
        ),
        backstory=(
            "You were fired from a high-frequency trading firm for using O(N^2) "
            "array scans instead of hash maps in an order-matching path, costing "
            "millions in missed fills. You now violently despise bloated code: "
            "nested loops over the same collection, repeated linear lookups, N+1 "
            "queries, needless copies and dead abstractions. You always cite the "
            "file and lines, never guess at code you have not read."
        ),
        tools=[outline_tool, search_tool, file_tool],
        llm=precise_llm,
        allow_delegation=False,
        max_iter=18,
        verbose=True,
    )

    synthesizer = Agent(
        role="Synthesizer - Actionable Prompt Author",
        goal=(
            "Write precise, actionable wording for every finding id so the report "
            "can be assembled into a prompt a developer can copy-paste into an "
            "LLM coding assistant."
        ),
        backstory=(
            "You were fired as a QA Lead for vague bug reports - 'it feels slow "
            "sometimes' was your last ticket. You are now obsessed with actionable "
            "prompt generation: every item names a file, a problem, a concrete fix "
            "and a way to verify it. You never drop a finding and never add one "
            "that the earlier agents did not report."
        ),
        llm=precise_llm,
        allow_delegation=False,
        max_iter=6,
        verbose=True,
    )

    # --- Tasks ---------------------------------------------------------------
    scout_task = Task(
        description=(
            f"Map the repository at {target_directory}.\n"
            "The stack below was detected deterministically from manifests, "
            "imports, script tags and config files. It is final: do not re-derive "
            "or contradict it. Copy primary_language, framework and "
            "stack_components from it.\n"
            "--- STACK FINGERPRINT ---\n"
            f"{fingerprint_block}\n"
            "--- END FINGERPRINT ---\n"
            "Your job is the part a scanner cannot do: what the software does for "
            "real users.\n"
            "1. Call the directory listing tool once.\n"
            "2. Read at most 6 files that reveal intent: README, main entry "
            "points, core domain logic.\n"
            "3. State the business intent and critical user flows, each backed by "
            "an exact quote.\n"
            f"{FILE_READ_RULES}\n"
            'If something is unknown, use the string "unknown" - do not guess.'
        ),
        expected_output=(
            "A single valid JSON object and nothing else (no prose, no markdown "
            "fences) with EXACTLY these keys:\n"
            "{\n"
            '  "repository_path": string,\n'
            '  "primary_language": string (copied from the fingerprint),\n'
            '  "framework": string (copied from the fingerprint),\n'
            '  "stack_components": array of strings (copied from the fingerprint),\n'
            '  "business_intent": string (2-4 sentences: who the users are, what '
            "they do with the product, and what going wrong would cost them),\n"
            '  "critical_user_flows": array of 3-6 strings,\n'
            '  "sensitive_data": array of strings (or empty array),\n'
            '  "entry_points": array of relative file paths,\n'
            '  "key_files": array of 5-15 relative file paths holding the core '
            "logic, ordered by importance,\n"
            '  "evidence": array of at least 2 objects, each with: "file" '
            "(relative path of a file you read with ReadRepoFileTool), "
            '"lines" (string, e.g. "3-9"), "quote" (1-6 lines copied exactly '
            "from that tool's output for this file, without the line-number "
            'prefixes), "fact" (string: what the quote proves)\n'
            "}"
        ),
        agent=scout,
        guardrail=scout_guardrail.as_task_guardrail(),
        guardrail_max_retries=scout_guardrail.crew_max_retries,
    )

    if judged_ids:
        inspector_instructions = (
            "A deterministic engine has already run the failure catalogue's "
            "rules over this repository. Its hits are listed below. Each starts "
            "with `<id> | <rule> | <title> | rule: <type>`, followed by "
            "`file:line | the verbatim code found there`.\n"
            "You are the judge of these hits, not a scanner. Do not search for "
            "other problems and do not add findings: an id that is not in this "
            "list is rejected.\n"
            "For EACH listed id decide, using the Scout's business intent:\n"
            '- "applies": the hit is a real problem for this product;\n'
            '- "not_applicable": a false positive here (say why);\n'
            '- "needs_review": cannot be decided from the evidence.\n'
            "You may call ReadRepoFileTool at most 6 times to read the lines "
            "around a hit before deciding.\n"
            "--- DETECTOR HITS ---\n"
            f"{hits_block}\n"
            "--- END HITS ---"
        )
    else:
        inspector_instructions = (
            "The deterministic detector engine ran the failure catalogue over "
            "this repository and reported no hits. There is nothing to judge. Do "
            'not look for findings yourself. Return {"judgments": []}.'
        )

    inspector_task = Task(
        description=inspector_instructions,
        expected_output=(
            "A single valid JSON object and nothing else with EXACTLY one key:\n"
            "{\n"
            '  "judgments": array with one object per listed id, each with: '
            '"id" (string, exactly as listed, e.g. "D3"), "verdict" ("applies" | '
            '"not_applicable" | "needs_review"), "severity" ("critical" | "high" '
            '| "medium" | "low"), "reason" (string: one or two sentences tying '
            "the quoted code to this product)\n"
            "}\n"
            "No other keys. No findings beyond the listed ids."
        ),
        agent=inspector,
        context=[scout_task],
        guardrail=inspector_guardrail.as_task_guardrail(),
        guardrail_max_retries=inspector_guardrail.crew_max_retries,
    )

    red_team_task = Task(
        description=(
            "Propose 3 to 5 extreme but physically plausible failure scenarios "
            "specific to THIS product that the Inspector's detector hits do not "
            "already cover. Think: dropped connections mid-write, double "
            "submits, clock skew, device rotation, concurrent edits, hostile "
            "input in the product's own domain, quota exhaustion.\n"
            f"Stack (deterministic fingerprint, final): {fingerprint.stack_trigger}\n"
            f"The only terms allowed in stack_trigger are: {allowed_stack_terms}\n"
            "Every scenario must be tied to code. Evidence is either a quote "
            "copied exactly from SearchRepoTool / ReadRepoFileTool output or "
            "from the Scout's evidence, or a detector hit id from the Inspector "
            '(e.g. {"detector": "D3"}). Use the tools to find the code that '
            "makes the scenario possible.\n"
            "For each scenario WITH evidence, call InsertDiscoveryTool exactly "
            "once with:\n"
            "- vulnerability_type: a short snake_case category\n"
            "- stack_trigger: comma-separated terms from the allowed list only\n"
            f"- binary_flags_dict: JSON object using these boolean keys: {flag_keys}\n"
            "- fix_recommendation: a concrete engineering fix (2-4 sentences)\n"
            "- confidence: 0 to 1\n"
            '- evidence: JSON array of {"file", "lines", "quote"} and/or '
            '{"detector": "<id>"} objects\n'
            "If the tool refuses the insert or returns status \"skipped\", do not "
            "retry more than once. A scenario you cannot back with evidence must "
            "NOT be inserted: report it with an empty evidence array and "
            "discovery_id null. It will be labelled a hypothesis."
        ),
        expected_output=(
            "A single valid JSON object and nothing else with EXACTLY one key:\n"
            "{\n"
            '  "scenarios": array of 3-5 objects, each with: "title" (string), '
            '"narrative" (string: 2-3 sentences on the real-world situation and '
            'the damage), "vulnerability_type" (string), "stack_trigger" (string, '
            'allowed terms only), "fix_recommendation" (string), "severity" '
            '("critical" | "high" | "medium" | "low"), "evidence" (array of '
            '{"file", "lines", "quote"} or {"detector": "<id>"} objects; empty '
            'if you have none), "discovery_id" (number returned by '
            "InsertDiscoveryTool, or null)\n"
            "}"
        ),
        agent=red_teamer,
        context=[scout_task, inspector_task],
        guardrail=red_team_guardrail.as_task_guardrail(),
        guardrail_max_retries=red_team_guardrail.crew_max_retries,
    )

    optimizer_task = Task(
        description=(
            "Audit the repository's code for bloat and algorithmic waste.\n"
            "1. Take the `key_files` and `entry_points` from the Scout's JSON and "
            "read up to 8 of the most logic-heavy source files with "
            "ReadRepoFileTool (OutlineFileTool and SearchRepoTool help you find "
            "the hot spots first).\n"
            "2. Hunt for: nested loops over the same or related collections, "
            "linear scans (`in list`, `.find`, `.includes`, `.index`) inside "
            "loops where a hash map/set belongs, N+1 database or network calls, "
            "repeated sorting, recomputation that should be cached, unbounded "
            "in-memory loads, needless copies, synchronous blocking I/O on hot "
            "paths, and dead or duplicated code.\n"
            "3. For each finding state the current and the target Big-O and the "
            "exact rewrite.\n"
            f"{FILE_READ_RULES}\n"
            "Only report issues in code you actually read; quote the offending "
            "lines. If a file is clean, list it under files_reviewed and move on."
        ),
        expected_output=(
            "A single valid JSON object and nothing else with EXACTLY these keys:\n"
            "{\n"
            '  "files_reviewed": array of relative file paths actually read,\n'
            '  "findings": array of objects, each with: '
            '"file" (relative path), "lines" (string, e.g. "42-57"), '
            '"symbol" (function/class name or "module level"), '
            '"issue" (string), "offending_code" (string: 1-8 lines copied exactly '
            "from ReadRepoFileTool's output for that file, without the "
            'line-number prefixes), "current_complexity" (string, e.g. "O(N^2) '
            'time, O(N) space"), "target_complexity" (string), "rewrite" '
            "(string: the concrete change, naming the data structure or query to "
            'use), "impact" ("critical" | "high" | "medium" | "low"),\n'
            '  "bloat": array of objects {"file": string, "lines": string, '
            '"what_to_delete_or_merge": string},\n'
            '  "summary": string (2-3 sentences on the overall efficiency of the '
            "codebase)\n"
            "}"
        ),
        agent=optimizer,
        context=[scout_task],
        guardrail=optimizer_guardrail.as_task_guardrail(),
        guardrail_max_retries=optimizer_guardrail.crew_max_retries,
    )

    synthesis_task = Task(
        description=(
            "The earlier outputs list this run's findings by id: D-ids are "
            "detector hits judged by the Inspector, V-ids are the Optimizer's "
            "verified findings, R-ids are the Red Teamer's scenarios. Each "
            "already carries an evidence label assigned by code.\n"
            "Write the wording for each finding id: the problem as it affects "
            "this product, the exact change required, and an acceptance check "
            "that proves the fix.\n"
            "You cannot add, merge, relabel or reorder findings. The final "
            "report is assembled by code from your wording: evidence-backed "
            "findings become work items, and hypotheses are placed in a "
            "separate final section. An id that is not in the earlier outputs is "
            "rejected. Do not state a hypothesis as fact in any field."
        ),
        expected_output=(
            "A single valid JSON object and nothing else with EXACTLY these keys:\n"
            "{\n"
            '  "role": string (one paragraph telling the coding assistant it is a '
            "senior engineer hardening this specific repository),\n"
            '  "items": array with one object per finding id, each with: "id" '
            '(string, exactly as given, e.g. "D3", "V1", "R2"), "problem" '
            '(string), "required_change" (string: specific instructions, '
            'including target Big-O for performance items), "acceptance_check" '
            "(string: a test or observable behaviour that proves the fix)\n"
            "}"
        ),
        agent=synthesizer,
        context=[scout_task, inspector_task, red_team_task, optimizer_task],
        guardrail=synthesizer_guardrail.as_task_guardrail(),
        guardrail_max_retries=synthesizer_guardrail.crew_max_retries,
    )

    # --- Resume: drop finished tasks and hand their outputs to the rest ------
    named_tasks = [
        ("scout", scout_task),
        ("inspector", inspector_task),
        ("red_team", red_team_task),
        ("optimizer", optimizer_task),
        ("synthesis", synthesis_task),
    ]
    if completed_outputs:
        name_of = {id(task): name for name, task in named_tasks}
        summary_tokens = budget_tokens(AGENT_SUMMARY_SHARE)
        for name, task in named_tasks:
            if name in completed_outputs:
                continue
            restored = [
                name_of[id(dep)] for dep in (task.context or []) if name_of[id(dep)] in completed_outputs
            ]
            if restored:
                task.context = [
                    dep for dep in task.context if name_of[id(dep)] not in completed_outputs
                ]
                task.description += "".join(
                    f"\n--- VERIFIED OUTPUT OF THE EARLIER {dep_name.upper()} TASK (restored from checkpoint) ---\n"
                    + truncate_to_tokens(completed_outputs[dep_name], summary_tokens)
                    for dep_name in restored
                )
    live_tasks = [task for name, task in named_tasks if name not in completed_outputs]
    live_agents = [task.agent for task in live_tasks]

    crew = Crew(
        agents=live_agents,
        tasks=live_tasks,
        process=Process.sequential,
        max_rpm=20,
        verbose=True,
    )
    return crew


def deterministic_report(run: RunContext) -> str:
    """The report built from deterministic results alone.

    Used when the LLM pipeline fails part-way: detector hits that no agent
    judged are registered as unjudged [Detector-Hit] findings, so the run
    still produces an evidence-backed report.
    """
    if not any(f.get("source") == "catalog" for f in run.findings.values()):
        InspectorGuardrail(run).on_accept({"judgments": []})
    return render_report(run, run.artifacts.get("synthesis"))


def _is_git_url(value: str) -> bool:
    return value.startswith(("https://", "http://", "git@"))


def _remove_tree(path: Path) -> None:
    """Delete a directory tree; git pack files are read-only on Windows."""

    def _retry_writable(func, failed_path, _exc):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    try:
        shutil.rmtree(path, onexc=_retry_writable)
    except OSError:
        pass


def _clone_repo(url: str) -> Path:
    """Shallow-clone a repository into a temp directory and return its path."""
    clone_root = Path(tempfile.mkdtemp(prefix="reporover_"))
    target = clone_root / "repo"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--", url, str(target)],
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _remove_tree(clone_root)
        raise RuntimeError(f"git clone failed for {url}: {exc}") from exc
    return target


def main(target: str | None = None, resume: bool = False) -> None:
    launch_directory = Path.cwd()
    target_directory = (
        target
        if target
        else input("Enter a GitHub URL or local folder path to analyze: ")
    ).strip().strip("\"'")
    if not target_directory:
        sys.exit("No repository URL or folder path provided.")

    cloned_dir = None
    result = None
    failure = None
    try:
        if _is_git_url(target_directory):
            cloned_dir = _clone_repo(target_directory)
            target_directory = str(cloned_dir)
        crew = run_analyzer(
            target_directory, resume=resume, checkpoint_directory=launch_directory
        )
        result = crew.kickoff()
    except Exception as exc:  # startup, LLM/provider or tool failure
        failure = exc
    finally:
        os.chdir(launch_directory)
        if cloned_dir is not None:
            _remove_tree(cloned_dir.parent)

    run = get_active_run()
    set_active_run(None)
    if run is None:
        sys.exit(f"RepoRover could not start: {failure}")

    if failure is None:
        report = str(result)
    else:
        # The deterministic passes already ran; report what they proved.
        report = deterministic_report(run)

    status = run.final_status(failed=failure is not None)
    run.status = status
    run_directory = launch_directory / RUNS_DIRECTORY / (
        f"{datetime.now():%Y%m%d_%H%M%S}_{run.run_uuid[:8]}"
    )
    ledger_path = run.save(run_directory)
    (run_directory / "findings.json").write_text(
        json.dumps(
            {
                "target_repo": None if cloned_dir is not None else str(run.root),
                "fingerprint": run.fingerprint.to_dict() if run.fingerprint else None,
                "catalog_source": run.artifacts.get("catalog_source"),
                "detector_findings": list(run.detector_findings.values()),
                "report_findings": list(run.findings.values()),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    finish_analysis_run(run.db_run_id, status, run.summary(), run.degraded_reasons)
    if failure is None and run.checkpoint is not None:
        run.checkpoint.mark_completed(run)

    banner = ""
    if failure is not None:
        banner = (
            "> **FAILED RUN** - the LLM pipeline stopped before finishing "
            f"({type(failure).__name__}: {failure}). This report contains the "
            "deterministic findings and whatever the agents had verified.\n\n"
        )
    elif run.degraded_reasons:
        banner = (
            "> **DEGRADED RUN** - some agent output failed evidence validation. "
            "Unverified claims were removed and database writes were disabled.\n"
            + "".join(f"> - {reason}\n" for reason in run.degraded_reasons)
            + "\n"
        )
    report = banner + report

    report_path = run_directory / "prompt.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"REPOROVER - COPY-PASTE PROMPT (run status: {status})")
    print("=" * 80 + "\n")
    print(report)
    print(f"\nSaved to {report_path}")
    print(f"Read ledger and findings saved to {run_directory}")

    # Chaos plan for the target, from evidence-backed findings only. Nothing is
    # executed here: running it needs `reporover chaos plan --run` and its gate.
    try:
        chaos_directory, chaos_plan = generate_plan(
            list(run.findings.values()),
            stack_trigger=run.fingerprint.stack_trigger if run.fingerprint else "unknown",
            target_repo=str(run.root),
            out_root=launch_directory / "out",
        )
        print(
            f"Chaos plan ({len(chaos_plan['experiments'])} experiment(s), not executed) "
            f"written to {chaos_directory}"
        )
    except OSError as exc:
        print(f"Warning: could not write the chaos plan: {exc}")
    if failure is not None:
        if cloned_dir is None and run.checkpoint is not None:
            print("Finished tasks are checkpointed. Continue with: reporover analyze <path> --resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
