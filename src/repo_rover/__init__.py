import os

# RepoRover analyses private source code, so nothing may leave the machine
# except the model calls the user configures. CrewAI's anonymous telemetry is
# therefore disabled unless the user explicitly turns it back on.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")  # never offer to upload execution traces


def main() -> None:
    from repo_rover.analysis_crew import main as run_analysis

    run_analysis()
