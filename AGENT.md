# Match Agent — instructions

Point your AI agent (with shell access on the STEACH box) at this file. Its job: match
church-service transcripts (STT session databases) against their YouTube captions and
produce **training-ready bundles**, then report the results.

## Step 1 — Ask the operator which dates
Before doing anything, ask:

> **"Which service dates should I process?"**

Accept a single date, a range, or a relative window, and convert to
`--since YYYY-MM-DD [--until YYYY-MM-DD]`:
- `2026-07-05` → `--since 2026-07-05 --until 2026-07-05`
- "first two weeks of July" → `--since 2026-07-01 --until 2026-07-14`
- "last 2 weeks" → compute the window from today's date.

## Step 2 — Match those dates from the STT backup (in place, read-only)
1. Ensure the STEACH server is up (it does the matching):
   - `curl -s http://localhost:5001/api/youtube/cache` — if it fails to connect,
     `cd /home/ai/STEACH && ./start.sh` and wait ~3 seconds.
2. Run the engine over the requested dates, reading DBs straight from the backup:
   ```bash
   cd /home/ai/STEACH
   python3 match_agent.py --backup ~/.stt/_AUTOMATIC_BACKUP \
       --since <FROM> [--until <TO>] --out match_output
   ```
   - The YouTube channel comes from server config; override with `--channel @handle`.
   - Add `--dry-run` first to show exactly which sessions will be processed.
   - It **skips** sessions already matched; add `--overwrite` to redo them.
   - It is **rate-limit aware** (HTTP 429 → waits and retries once). If you see many
     `[429]` lines, wait a few minutes and re-run — done sessions are reused from cache.

## Step 3 — Read and report the log
Each session prints one line:
- `[ ok ] … sim=0.80 kept=414 …` — matched. `sim` (0–1) is alignment quality (higher is
  better); `kept` is the number of training rows.
- `[none] …` — that service has **no YouTube captions** (normal; nothing to train from).
- `[fail] …` — an error; include the message in your report.

Report back: how many matched, their `sim`/`kept`, which were `[none]`, any `[fail]`s,
and confirm the bundles are in `match_output/`.

## Notes
- DBs are read **in place** from the backup, so each session's companion `.wav` stays
  next to its `.db` — that `.wav` is needed later when building training audio clips.
- Bundles in `match_output/` are ready to **upload → review → train** in the STEACH web
  UI ("⤒ Upload bundle to server"). Matching also populates the server cache, so those
  sessions can be reviewed in the UI directly (offline, no re-fetch).
- Captions are always the **original language** (e.g. Russian `ru-orig`), never an
  English translation.
- Treat `~/.stt/_AUTOMATIC_BACKUP` as **read-only**. Only write under `match_output/`.
  Don't copy large `.wav` files around or commit anything.

## Success criteria
Every session in the requested date range ends up with either a bundle in
`match_output/` or a clear `[none]` / `[fail]` explanation in your report.
