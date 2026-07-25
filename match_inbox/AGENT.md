# Match Agent — instructions

You are an assistant with shell access on this machine (the STEACH box). This folder
(`match_inbox/`) holds **STT session database files** (`.db`) from church services.
Your job: match each session's transcript against its YouTube service captions and
produce **training-ready bundles**, then report what happened.

## What "matching" does (handled for you by the tooling)
For each `.db`: resolve the matching YouTube video by the session's date, download the
**original-language** auto-captions (e.g. Russian `ru-orig`, never an English
translation), time-align them to the DB rows (anchor-based, tracks clock drift), skip
rows the STT app already rejected (denied / partial), and score alignment quality.
Output = one bundle zip per session (session DB + captions + `alignment.json`) written
to `../match_output/`.

## Steps
1. **Ensure the STEACH server is up** (it does the actual work):
   - Check: `curl -s http://localhost:5001/api/youtube/cache`
   - If that fails to connect: `cd /home/ai/STEACH && ./start.sh` then wait ~3 seconds.
2. **Run the matcher over this folder**:
   ```bash
   cd /home/ai/STEACH
   python3 match_agent.py --backup match_inbox --out match_output
   ```
   - The YouTube channel comes from the server config. Override with `--channel @handle`
     if needed.
   - It **skips** sessions that already have a bundle; add `--overwrite` to redo them.
   - It is **rate-limit aware**: on HTTP 429 it waits and retries once. If you see many
     `[429]` lines, wait a few minutes and re-run — already-fetched sessions are
     reprocessed offline from cache.
3. **Read the per-session log lines**:
   - `[ ok ]  … sim=0.80 kept=414 …` — matched. `sim` (0–1) is mean alignment
     agreement; higher is better. `kept` is the number of training rows.
   - `[none]  …` — that service has **no YouTube captions** (nothing to train from; this
     is normal for some services).
   - `[fail]  …` — an error; include the message in your report.
4. **Report back**: how many matched, their `sim`/`kept`, which were `[none]`, any
   `[fail]`s, and confirm the bundles are in `match_output/`.

## Notes
- Only the `.db` is needed to **match**. To also build training **audio clips** later,
  the session's companion `.wav` must sit next to the `.db` — so copy the `.wav` in too
  if you want clips, not just alignment review.
- Bundles in `match_output/` are ready to **upload → review → train** in the STEACH web
  UI ("⤒ Upload bundle to server"). Matching also populates the server cache, so those
  sessions can be reviewed in the UI directly (offline, no re-fetch).
- Treat the `.db`/`.wav` files here as **read-only inputs**. Only write under
  `match_output/`. Don't commit anything.

## Success criteria
Every `.db` in this folder ends up with either a bundle in `match_output/` or a clear
`[none]` / `[fail]` explanation in your report.
