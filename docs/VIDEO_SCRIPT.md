# RepoRover: 2-minute project video script

Length: about 2:00 (roughly 290 spoken words). Terminal only, dark theme, font large enough to read
at 720p. Everything shown is real output of the commands listed; run them once beforehand so
windows are sized and nothing scrolls unexpectedly.

Setup before recording
- Clear the screen and delete earlier demo output: `reporover_runs/demo_*` and `out/static_ledger`.
- Use `reporover demo --pace 2` so each stage pauses for two seconds.
- The demo is a replay of a real recorded model run. Its banner says so on screen; do not describe
  it as live.

| Time | [Visual Action on Screen] | [Voiceover / Text] |
|---|---|---|
| 0:00-0:08 | Empty terminal. Type `reporover --help`; the command list appears: analyze, catalog, chaos, bench, demo. | "AI-written apps ship fast, and ship flaws nobody asked about. RepoRover is an agentic reviewer that can only report what it can prove." |
| 0:08-0:14 | Type `reporover demo --pace 2`. Banner: `fixture: bench/fixtures/static_ledger | REPLAY of recorded traffic - offline`. | "This is a recorded run replayed offline: no network, no API key. The pipeline itself is running for real." |
| 0:14-0:24 | Stage `[1/8] Deterministic fingerprint (no LLM)`. Lines: `Alpine.js [framework] via script_tag at index.html:8`, `localStorage [platform_api] via api_usage at index.html:35`. | "Step one uses no model. The stack is fingerprinted from script tags, manifests and imports, each with the line that proves it." |
| 0:24-0:36 | Stage `[2/8] Detector engine: 16 catalogue rules fired (no LLM)`. Rows `D1 imported-data-not-validated index.html:62` through `D8 storage-quota-not-handled index.html:40`. | "Step two, still no model. A catalogue of 143 failure modes runs over the code. Sixteen rules fire, each pinned to a file and line." |
| 0:36-0:50 | Stage `[3/8] Scout`. Highlight `read ledger: 2 file(s), 68 line(s) served to agents` and `outputs rejected by guardrails: 2`. | "Now the agents. The Scout reads the code through confined tools, and every line it sees goes into a read ledger. Twice its answer was rejected, because a claim did not match the ledger. It had to retry." |
| 0:50-1:02 | Stage `[4/8] Inspector`. Highlight `findings so far: 0 Verified, 13 Detector-Hit, 0 Hypothesis`. | "The Inspector does not search for bugs. It judges the deterministic hits, and it cannot add one of its own. Thirteen stand." |
| 1:02-1:14 | Stages `[5/8] Red Teamer`, `[6/8] Optimizer`, `[7/8] Synthesizer` scroll past. Highlight `2 Verified, 14 Detector-Hit, 0 Hypothesis`. | "The Red Teamer adds two scenarios, each backed by an exact quote. The Synthesizer writes the wording, but labels are assigned by code." |
| 1:14-1:28 | Stage `[8/8] Chaos plan generated`. Rows `C01 playwright playwright/storage-quota.spec.js <- D4, D8`, `C02 ... double-submit.spec.js <- D9`, `C03 ... multi-tab-overwrite.spec.js <- D7`, `C04 ... malformed-import.spec.js <- D1`. | "Then it goes from finding to proof. Four Playwright experiments are generated, each tied to the finding that justifies it." |
| 1:28-1:40 | Open `out/static_ledger/chaos/playwright/storage-quota.spec.js`. Show the header (`Findings: D4 ..., D8 ...`, `Hypothesis: ...`), the `Refusing to run: BASE_URL ... is not a local sandbox` guard, and the fill loop. | "This one fills the browser's storage quota, then checks that no saved record is lost. It refuses to run against anything but a local sandbox." |
| 1:40-1:48 | Back in the terminal. Final summary lines: `Run status: completed | LLM calls: 20 | guardrail rejections: 3`, `Evidence-backed findings failing independent re-verification: 0`, `Replay matches the recorded run: yes`. | "Twenty model calls, three rejected by guardrails, and no finding that fails re-verification." |
| 1:48-1:56 | Type `reporover chaos self`. Eleven `[PASSED]` lines scroll; end on `Summary: 11 passed`. | "We hold ourselves to the same standard. Rate limits, outages, timeouts and malformed output are injected into RepoRover itself. Eleven of eleven pass." |
| 1:56-2:04 | Type `reporover bench`. Table ends `OVERALL ... 90.0% 85.1% 0.875` and `Hallucination rate: 0.0% (0/64 ...)`. Hold on this frame. | "On our fixtures: ninety percent precision, eighty-five percent recall, zero unverifiable citations. RepoRover: evidence first." |

## Notes for the recording

- **On-screen text to add in the edit (optional):** "Replay of a recorded run on a synthetic
  fixture" during 0:08-1:48, and "Benchmark: 3 synthetic fixtures, detector stage" on the last shot.
  These keep the video accurate when viewed without sound.
- **If a stage scrolls too fast,** raise `--pace`; the replay itself takes under a second.
- **CrewAI prints one "Crew Execution Started" box** after stage 2. Leave it in or cut it in the edit.
- **Do not show** `.env`, the Neon console, or any window containing a key.
- **Numbers spoken in the script** come from `reporover demo`, `reporover chaos self` and
  `reporover bench` as of the last run. If you re-record the cassette or change the fixtures,
  re-check them before recording the voiceover.
- **To run inside the 1-4 minute limit with margin,** the table above totals about 2:04. Dropping
  the 1:28-1:40 shot brings it to about 1:50.
