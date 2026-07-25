#!/usr/bin/env python3
"""STEACH match agent — batch-match STT session DBs to YouTube captions.

Walks a backup folder for session `.db` files in a date range and, for each, drives
the running STEACH server's /api/youtube/export endpoint (which resolves the video by
date, downloads the original-language captions, aligns them to the DB rows, honors the
DB's own denied/partial flags, and applies the similarity gate). It saves one debug
bundle zip per session — DB + captions + alignment.json — into an output folder, ready
to upload/review/train in STEACH.

The heavy matching runs server-side (and populates its cache), so this stays a thin,
dependency-free driver: standard library only, rate-limit aware, resumable.

Usage:
    ./match_agent.py --backup /home/ai/.stt/_AUTOMATIC_BACKUP \
                     --since 2026-07-01 [--until 2026-07-31] \
                     --out ./matched [--channel @bcoscharlotte] \
                     [--server http://localhost:5001] [--sleep 5] [--overwrite] [--dry-run]

Session dates come from the DB filename stem (YYYY-MM-DD...). If --channel is omitted
it is read from the server's config. Already-produced bundles are skipped unless
--overwrite. On HTTP 429 (YouTube rate-limit) the agent backs off and retries once.
"""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


def log(msg):
    print(msg, flush=True)


def parse_stem_date(stem):
    """Extract a date from a session filename stem like '2026-07-05_093218'."""
    try:
        return datetime.strptime(stem[:10], "%Y-%m-%d").date()
    except (ValueError, IndexError):
        return None


def discover_dbs(backup_dir, since, until):
    """Yield (path, date) for session .db files in [since, until], sorted by date."""
    found = []
    for p in Path(backup_dir).expanduser().rglob("*.db"):
        if not p.is_file():
            continue
        d = parse_stem_date(p.stem)
        if since and (d is None or d < since):
            continue
        if until and (d is None or d > until):
            continue
        found.append((p, d))
    found.sort(key=lambda x: (x[1] or datetime.min.date(), str(x[0])))
    return found


def _post_json(url, body, timeout, expect_zip):
    """POST JSON; return (ok, content_type, bytes_or_dict). Never raises for HTTP errors."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read()
            if expect_zip and "application/zip" in ctype:
                return True, ctype, data
            try:
                return True, ctype, json.loads(data.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                return True, ctype, data
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            return False, "", json.loads(body_bytes.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return False, "", {"error": f"HTTP {e.code}: {body_bytes[:200]!r}"}
    except urllib.error.URLError as e:
        return False, "", {"error": f"cannot reach server: {e.reason}"}


def get_default_channel(server, timeout):
    try:
        with urllib.request.urlopen(f"{server}/api/config", timeout=timeout) as resp:
            cfg = json.loads(resp.read().decode("utf-8", "replace"))
        rem = cfg.get("remember", {}) or {}
        return rem.get("channel") or (cfg.get("youtube", {}) or {}).get("channel") or ""
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return ""


def bundle_summary(zip_bytes):
    """Read alignment.json out of a bundle for a one-line log."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            meta = json.loads(z.read("alignment.json").decode("utf-8", "replace"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        return ""
    rep = meta.get("report", {}) or {}
    return (f"video={meta.get('video')} lang={meta.get('caption_lang')} "
            f"anchors={meta.get('anchors')} kept={len(meta.get('kept', []))} "
            f"sim={rep.get('mean_similarity')} drops={meta.get('drop_summary')}")


def process_one(server, channel, db_path, out_zip, timeout):
    """Export one session's bundle. Returns ('ok'|'skip'|'nocap'|'ratelimit'|'error', detail)."""
    ok, _ctype, payload = _post_json(
        f"{server}/api/youtube/export",
        {"db_path": str(Path(db_path).resolve()), "channel": channel},
        timeout=timeout, expect_zip=True,
    )
    if ok and isinstance(payload, (bytes, bytearray)):
        out_zip.write_bytes(payload)
        return "ok", bundle_summary(payload)
    err = (payload or {}).get("error", "unknown error") if isinstance(payload, dict) else str(payload)
    blob = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    if "429" in blob:
        return "ratelimit", err
    if "no usable auto-captions" in err or "no channel video matched" in err:
        return "nocap", err
    return "error", err


def main():
    ap = argparse.ArgumentParser(description="Batch-match STT session DBs to YouTube captions.")
    ap.add_argument("--backup", required=True, help="root folder holding session .db files")
    ap.add_argument("--out", required=True, help="output folder for bundle zips")
    ap.add_argument("--since", help="only sessions on/after this date (YYYY-MM-DD)")
    ap.add_argument("--until", help="only sessions on/before this date (YYYY-MM-DD)")
    ap.add_argument("--channel", help="YouTube channel (default: from server config)")
    ap.add_argument("--server", default="http://localhost:5001", help="STEACH server URL")
    ap.add_argument("--sleep", type=float, default=5.0, help="seconds to wait between sessions")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    ap.add_argument("--overwrite", action="store_true", help="re-match sessions that already have a bundle")
    ap.add_argument("--dry-run", action="store_true", help="list what would be processed, do nothing")
    args = ap.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d").date() if args.since else None
    until = datetime.strptime(args.until, "%Y-%m-%d").date() if args.until else None
    server = args.server.rstrip("/")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel = args.channel or get_default_channel(server, args.timeout)
    if not channel:
        log("ERROR: no --channel given and none configured on the server.")
        return 2

    sessions = discover_dbs(args.backup, since, until)
    if not sessions:
        log("No session .db files matched the date range.")
        return 0

    rng = f"{since or 'start'}..{until or 'end'}"
    log(f"Found {len(sessions)} session(s) in {rng} | channel={channel} | server={server}")
    if args.dry_run:
        for p, d in sessions:
            log(f"  would match {d} {p.name}")
        return 0

    counts = {"ok": 0, "skip": 0, "nocap": 0, "ratelimit": 0, "error": 0}
    for p, d in sessions:
        out_zip = out_dir / f"steach_debug_{p.stem}.zip"
        if out_zip.exists() and not args.overwrite:
            log(f"[skip] {p.name} (bundle exists)")
            counts["skip"] += 1
            continue

        status, detail = process_one(server, channel, p, out_zip, args.timeout)
        if status == "ratelimit":
            log(f"[429 ] {p.name} — backing off 60s, then one retry")
            time.sleep(60)
            status, detail = process_one(server, channel, p, out_zip, args.timeout)

        counts[status if status in counts else "error"] += 1
        tag = {"ok": "[ ok ]", "nocap": "[none]", "error": "[fail]", "ratelimit": "[429 ]"}.get(status, "[fail]")
        log(f"{tag} {d} {p.name} — {detail}")

        if args.sleep > 0 and p is not sessions[-1][0]:
            time.sleep(args.sleep)

    log(f"\nDone. matched={counts['ok']} no-captions={counts['nocap']} "
        f"skipped={counts['skip']} rate-limited={counts['ratelimit']} errors={counts['error']}")
    log(f"Bundles in {out_dir}/ — upload them into STEACH to review and train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
