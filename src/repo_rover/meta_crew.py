from dotenv import load_dotenv

load_dotenv()

import os
import sys
import time
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai_tools import DirectoryReadTool, FileReadTool, ScrapeWebsiteTool

from repo_rover.db_tools import InsertDiscoveryTool
from repo_rover.runtime import (
    BoundedScrapeWebsiteTool,
    GroqChatLLM,
    RepoDirectoryReadTool,
    build_chat,
    require_env,
)

REPO_ROVER_SOURCE_DIR = Path(__file__).resolve().parent

DEFAULT_INTEL_SOURCES = (
    "https://github.com/advisories?query=type%3Areviewed+ecosystem%3Anpm",
    "https://github.com/advisories?query=type%3Areviewed+ecosystem%3Apip",
    "https://github.com/vercel/next.js/security/advisories",
    "https://nodejs.org/en/blog/vulnerability",
)

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


def _intel_sources() -> list[str]:
    """Sources to scrape: META_INTEL_SOURCES (comma-separated URLs) or the defaults."""
    configured = os.getenv("META_INTEL_SOURCES", "")
    sources = [url.strip() for url in configured.split(",") if url.strip()]
    return sources or list(DEFAULT_INTEL_SOURCES)


def run_meta_crew():
    """Run one pass of the background meta crew and return its result.

    Never touches a user's repository: Agent 6 reads public web sources and
    writes to `dynamic_discoveries`; Agent 7 reads RepoRover's own source.
    """
    require_env("LLM_API_KEY")
    require_env("DATABASE_URL")

    launch_directory = Path.cwd()
    source_directory = str(REPO_ROVER_SOURCE_DIR)
    # crewai_tools' DirectoryReadTool only lists paths inside the working directory.
    os.chdir(source_directory)

    try:
        chat = build_chat(0.1)
        llm = GroqChatLLM(chat)

        scrape_tool: ScrapeWebsiteTool = BoundedScrapeWebsiteTool()
        directory_tool: DirectoryReadTool = RepoDirectoryReadTool(
            directory=source_directory
        )
        file_tool = FileReadTool(base_dir=source_directory)

        sources = _intel_sources()
        source_list = "\n".join(f"- {url}" for url in sources)
        flag_keys = ", ".join(BINARY_FLAG_KEYS)

        market_scraper = Agent(
            role="Market Intelligence Scraper",
            goal=(
                "Scour web documentation and GitHub issues/advisories for newly "
                "discovered vulnerabilities and push every one of them into the "
                "database."
            ),
            backstory=(
                "You were fired for missing a critical zero-day patch in a React "
                "dependency - the advisory had been public for nine days while "
                "production stayed exposed. You now obsessively seek out new "
                "failure nodes and cannot rest until each one is inserted into "
                "the database. You only record what a source actually says."
            ),
            tools=[scrape_tool, InsertDiscoveryTool],
            llm=llm,
            allow_delegation=False,
            max_iter=22,
            verbose=True,
        )

        self_auditor = Agent(
            role="Self-Auditor",
            goal=(
                "Audit the Python code of this very RepoRover tool for "
                "inefficiencies and robustness gaps."
            ),
            backstory=(
                "You were fired because the security software you built crashed "
                "itself in production while scanning a customer. You now "
                "constantly check our own codebase for robustness: unhandled "
                "errors, leaked connections, unbounded inputs, wasted LLM calls "
                "and anything that could make RepoRover fall over mid-run. You "
                "only report on code you have actually read."
            ),
            tools=[directory_tool, file_tool],
            llm=llm,
            allow_delegation=False,
            max_iter=16,
            verbose=True,
        )

        intel_task = Task(
            description=(
                "Collect newly published vulnerabilities from these sources:\n"
                f"{source_list}\n"
                "1. Scrape each URL with the website reading tool (pass it as "
                "`website_url`). If a page fails to load, record it and continue.\n"
                "2. From the scraped text, extract up to 8 distinct, recent "
                "vulnerabilities or failure modes in total. Each must name an "
                "affected package/framework and what goes wrong.\n"
                "3. For EVERY extracted vulnerability you MUST call "
                "InsertDiscoveryTool exactly once to log it to the "
                "`dynamic_discoveries` table, with:\n"
                "- vulnerability_type: short snake_case category (e.g. "
                '"prototype_pollution", "ssrf", "path_traversal")\n'
                "- stack_trigger: the affected package/framework and version "
                "range, as the source states it\n"
                "- binary_flags_dict: a JSON object using these boolean keys: "
                f"{flag_keys}\n"
                "- fix_recommendation: the patched version or mitigation, plus "
                "the advisory identifier (GHSA/CVE) and source URL\n"
                "- confidence: 0 to 1, how clearly the source states the issue\n"
                "- evidence: a JSON array with one object {\"source_url\": the "
                "https URL you scraped, \"quote\": a sentence copied from that "
                "page that states the vulnerability}\n"
                "A finding that was not inserted with InsertDiscoveryTool does "
                "not count. Do not invent advisories, identifiers or versions."
            ),
            expected_output=(
                "A single valid JSON object and nothing else with EXACTLY these "
                "keys:\n"
                "{\n"
                '  "sources_scraped": array of objects {"url": string, "status": '
                '"ok" | "failed: <reason>"},\n'
                '  "discoveries": array of objects, each with: '
                '"advisory_id" (GHSA/CVE id or "none stated"), "source_url" '
                '(string), "vulnerability_type" (string), "stack_trigger" '
                '(string), "binary_flags" (object of booleans), '
                '"fix_recommendation" (string), "discovery_id" (number returned '
                "by InsertDiscoveryTool, or null if the insert failed),\n"
                '  "inserted_count": number of rows written to '
                "dynamic_discoveries,\n"
                '  "insert_errors": array of strings (empty if none)\n'
                "}"
            ),
            agent=market_scraper,
        )

        audit_task = Task(
            description=(
                f"Audit RepoRover's own source code in {source_directory}.\n"
                "1. Call the directory listing tool once.\n"
                "2. Read every .py file it lists with the file reading tool, "
                "passing the listed relative path as file_path and line_count=200 "
                "(page with start_line for longer files).\n"
                "3. Look for: unhandled exceptions that would crash a run, "
                "database connections that are not closed or reused wastefully, "
                "unbounded tool output fed to the LLM, missing timeouts and "
                "retries, redundant LLM or database calls, O(N^2) loops, "
                "duplicated code across modules, and weak input validation.\n"
                "Quote the lines you are criticising. Do not report problems in "
                "code you did not read."
            ),
            expected_output=(
                "A markdown report with these sections:\n"
                "1. `## Files Audited` - bullet list of relative paths read.\n"
                "2. `## Findings` - a numbered list ordered by severity; each "
                "item has **File** and **Lines**, **Category** (robustness | "
                "performance | cost | maintainability), **Severity** (critical | "
                "high | medium | low), **Problem** with the quoted code, and "
                "**Fix** with the concrete code-level change.\n"
                "3. `## Complexity Notes` - any function whose time or space "
                "complexity is worse than necessary, with current and target "
                "Big-O, or the sentence 'No complexity issues found.'\n"
                "4. `## Verdict` - two sentences on whether RepoRover is safe to "
                "leave running unattended."
            ),
            agent=self_auditor,
        )

        crew = Crew(
            agents=[market_scraper, self_auditor],
            tasks=[intel_task, audit_task],
            process=Process.sequential,
            max_rpm=20,
            verbose=True,
        )
        return crew.kickoff()
    finally:
        os.chdir(launch_directory)


def main() -> None:
    """Run the meta crew once, or continuously if META_LOOP_INTERVAL_SECONDS is set."""
    interval_raw = os.getenv("META_LOOP_INTERVAL_SECONDS", "").strip()
    try:
        interval = int(interval_raw) if interval_raw else 0
    except ValueError:
        sys.exit("META_LOOP_INTERVAL_SECONDS must be a whole number of seconds.")

    while True:
        try:
            result = run_meta_crew()
        except RuntimeError as exc:
            sys.exit(f"RepoRover meta crew could not start: {exc}")
        except Exception as exc:
            if interval <= 0:
                raise
            print(f"Meta crew pass failed, will retry next cycle: {exc}")
        else:
            print("\n" + "=" * 80)
            print("REPOROVER META CREW - SELF-AUDIT REPORT")
            print("=" * 80 + "\n")
            print(result)

        if interval <= 0:
            break
        print(f"\nNext meta crew pass in {interval} seconds. Press Ctrl+C to stop.")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("Meta loop stopped.")
            break


if __name__ == "__main__":
    main()
