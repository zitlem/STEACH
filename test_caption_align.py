"""Unit tests for caption_align — deterministic, no network, no yt-dlp."""

import caption_align as ca


# ---------------------------------------------------------------------------
# parse_vtt / to_word_stream
# ---------------------------------------------------------------------------

PLAIN_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.000 --> 00:00:05.000
this is a test
"""

# yt-dlp auto-sub style: rolling repeats with inline word timings + <c> tags.
AUTOSUB_VTT = """WEBVTT

00:00:00.000 --> 00:00:02.000
hello

00:00:00.000 --> 00:00:04.000
hello<00:00:02.000><c> world</c>

00:00:04.000 --> 00:00:06.000 align:start position:0%
hello world<00:00:05.000><c> again</c>
"""


def test_parse_vtt_plain():
    cues = ca.parse_vtt(PLAIN_VTT)
    assert len(cues) == 2
    assert cues[0].start_s == 1.0 and cues[0].end_s == 3.0
    assert cues[0].text == "Hello world"
    assert cues[1].text == "this is a test"


def test_parse_vtt_ignores_header_and_settings():
    cues = ca.parse_vtt(AUTOSUB_VTT)
    # Three timing lines -> three cues; the "align:start position:0%" settings
    # on the last timing line must not break parsing.
    assert len(cues) == 3
    assert cues[2].start_s == 4.0


def test_to_word_stream_dedupes_rolling_repeats():
    cues = ca.parse_vtt(AUTOSUB_VTT)
    stream = ca.to_word_stream(cues)
    words = [w["word"] for w in stream]
    # "hello" (t=0), "world" (t=2), "again" (t=5) — each once despite repeats.
    assert words == ["hello", "world", "again"]
    times = [w["t_s"] for w in stream]
    assert times == sorted(times)
    assert stream[1]["word"] == "world" and stream[1]["t_s"] == 2.0
    assert stream[2]["t_s"] == 5.0


def test_to_word_stream_strips_c_tags_and_normalizes():
    cues = ca.parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n<c>Hello,</c> World!\n")
    stream = ca.to_word_stream(cues)
    assert [w["word"] for w in stream] == ["hello", "world"]


# ---------------------------------------------------------------------------
# estimate_offset
# ---------------------------------------------------------------------------

def _rows(*triples):
    """Build DB rows from (text, start, end) triples."""
    out = []
    for i, (text, s, e) in enumerate(triples):
        out.append({"id": i + 1, "text": text, "start_time": s, "end_time": e,
                    "timestamp": "2026-05-10 18:30:00"})
    return out


def test_estimate_offset_recovers_known_shift():
    rows = _rows(
        ("the quick brown fox", 0.0, 2.0),
        ("jumps over the lazy dog", 2.0, 4.0),
    )
    # Captions are the same content shifted +30s on the YouTube clock.
    shift = 30.0
    db_stream = ca._db_word_stream(rows)
    caption_words = [{"word": w, "t_s": t + shift} for w, t in db_stream]
    offset = ca.estimate_offset(rows, caption_words)
    assert abs(offset - shift) < 0.5


def test_estimate_offset_no_overlap_returns_zero():
    rows = _rows(("alpha beta gamma", 0.0, 2.0))
    caption_words = [{"word": "totally", "t_s": 5.0}, {"word": "different", "t_s": 6.0}]
    assert ca.estimate_offset(rows, caption_words) == 0.0


# ---------------------------------------------------------------------------
# anchor-based alignment + VTT entity decoding
# ---------------------------------------------------------------------------

def test_parse_vtt_unescapes_html_entities():
    # YouTube speaker markers ">>" arrive as &gt;&gt; — must not become "gt"/"gtgt".
    cues = ca.parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n&gt;&gt; hello &amp; world\n")
    words = [w["word"] for w in ca.to_word_stream(cues)]
    assert words == ["hello", "world"]


def test_map_caption_time_interpolates_and_extrapolates():
    anchors = [(10.0, 110.0), (20.0, 140.0)]  # slope 3 between anchors
    assert ca.map_caption_time(anchors, 15.0) == 125.0   # interpolate
    assert ca.map_caption_time(anchors, 5.0) == 105.0    # extrapolate before (offset +100)
    assert ca.map_caption_time(anchors, 25.0) == 145.0   # extrapolate after (offset +120)
    assert ca.map_caption_time([], 42.0) == 42.0         # identity with no anchors


def test_anchored_alignment_tracks_clock_drift():
    rows = _rows(
        ("alpha bravo charlie", 0.0, 3.0),
        ("delta echo foxtrot", 3.0, 6.0),
        ("golf hotel india", 100.0, 103.0),
        ("juliet kilo lima", 103.0, 106.0),
    )
    # Captions drift mid-service: first block +1000s, second block +2000s.
    words = [{"word": w, "t_s": t + (1000.0 if t < 50 else 2000.0)}
             for w, t in ca._db_word_stream(rows)]

    anchors = ca.build_anchors(rows, words)
    assert len(anchors) >= 6
    anchored = ca.label_rows_anchored(rows, words, anchors)
    for lb in anchored:
        assert lb.similarity == 1.0, (lb.db_text, lb.corrected_text)

    # A single global offset cannot span both blocks.
    single = ca.estimate_offset(rows, words)
    single_labels = ca.label_rows(rows, words, single)
    assert sum(l.similarity for l in single_labels) < sum(l.similarity for l in anchored)


def test_refine_span_trims_over_capture_and_drift():
    db = "what do we do today".split()
    # caption window bleeds into the next sentence on both sides
    cand = "and now what do we do today do we follow people".split()
    refined = ca._refine_span(db, cand)
    assert refined == ["what", "do", "we", "do", "today"]
    # nothing in common -> empty (row will be filtered, not mislabeled)
    assert ca._refine_span(["amen"], ["completely", "different", "words"]) == []


def test_anchored_labeling_is_content_aware():
    # Row's audio is one short phrase; the caption window (wide) also contains the
    # next sentence. Content refinement must keep only the row's phrase.
    rows = _rows(("the lord is good", 0.0, 3.0), ("sing to him", 3.0, 6.0))
    words = [
        {"word": "the", "t_s": 0.4}, {"word": "lord", "t_s": 0.9},
        {"word": "is", "t_s": 1.4}, {"word": "good", "t_s": 2.2},
        {"word": "sing", "t_s": 3.3}, {"word": "to", "t_s": 3.8}, {"word": "him", "t_s": 4.4},
    ]
    anchors = ca.build_anchors(rows, words)
    labels = ca.label_rows_anchored(rows, words, anchors, pad_s=3.0)
    assert labels[0].corrected_text == "the lord is good"   # not "...good sing to him"
    assert labels[1].corrected_text == "sing to him"
    assert labels[0].similarity == 1.0 and labels[1].similarity == 1.0


def test_build_anchors_drops_backwards_pairs():
    # Two shared distinctive words, but the second occurs earlier in captions than
    # the first — a monotonic map can keep only one of them.
    rows = _rows(("alpha bravo", 0.0, 4.0))
    words = [{"word": "alpha", "t_s": 100.0}, {"word": "bravo", "t_s": 50.0}]
    anchors = ca.build_anchors(rows, words)
    assert len(anchors) == 1


# ---------------------------------------------------------------------------
# label_rows
# ---------------------------------------------------------------------------

def test_label_rows_attributes_by_time_overlap():
    rows = _rows(
        ("hello wrold", 0.0, 2.0),   # DB row has a typo the captions fix
        ("second line here", 2.0, 4.0),
    )
    caption_words = [
        {"word": "hello", "t_s": 0.5},
        {"word": "world", "t_s": 1.5},
        {"word": "second", "t_s": 2.5},
        {"word": "line", "t_s": 3.0},
        {"word": "here", "t_s": 3.5},
    ]
    labels = ca.label_rows(rows, caption_words, offset=0.0)
    assert labels[0].corrected_text == "hello world"
    assert labels[1].corrected_text == "second line here"
    # Row 1 got the "world" fix, so similarity to the typo'd DB text is < perfect.
    assert labels[0].similarity < 1.0
    assert labels[1].similarity == 1.0


def test_label_rows_respects_offset():
    rows = _rows(("hello world", 0.0, 2.0))
    # Captions on a clock +10s ahead.
    caption_words = [{"word": "hello", "t_s": 10.5}, {"word": "world", "t_s": 11.5}]
    labels = ca.label_rows(rows, caption_words, offset=10.0)
    assert labels[0].corrected_text == "hello world"


# ---------------------------------------------------------------------------
# filter_labels
# ---------------------------------------------------------------------------

def _label(db_text, corrected):
    return ca.RowLabel(
        transcription_id=1, start_time=0.0, end_time=1.0,
        db_text=db_text, corrected_text=corrected,
        similarity=ca._similarity(db_text, corrected),
    )


def test_filter_labels_keeps_good_drops_junk():
    good = _label("hello world", "hello world")
    empty = _label("hello world", "")
    halluc = _label("please give", "please subscribe to the channel")
    mismatch = _label("one", "one two three four five six seven")
    kept, dropped = ca.filter_labels([good, empty, halluc, mismatch])
    assert kept == [good]
    reasons = {lb.drop_reason for lb in dropped}
    assert reasons == {"empty", "hallucination", "length_mismatch"}


def test_rowlabel_to_dict_roundtrips_fields():
    lb = _label("hello wrld", "hello world")
    lb.drop_reason = None
    d = lb.to_dict()
    assert d["db_text"] == "hello wrld"
    assert d["corrected_text"] == "hello world"
    assert d["transcription_id"] == 1
    assert 0.0 <= d["similarity"] <= 1.0
    assert d["drop_reason"] is None


def test_filter_labels_allows_moderate_length_difference():
    lb = _label("the lord is good", "the lord is very good indeed")  # 4 vs 6 words
    kept, dropped = ca.filter_labels([lb])
    assert kept == [lb] and not dropped


# ---------------------------------------------------------------------------
# match_report
# ---------------------------------------------------------------------------

def test_match_report_high_for_aligned():
    kept = [_label("hello world", "hello world"), _label("good morning", "good morning")]
    report = ca.match_report(kept)
    assert report["row_count"] == 2
    assert report["mean_similarity"] == 1.0
    assert report["low_match_count"] == 0


def test_match_report_flags_low_matches():
    kept = [_label("hello world", "hello world"),
            _label("alpha beta gamma", "totally unrelated text")]
    report = ca.match_report(kept)
    assert report["low_match_count"] == 1
    assert report["mean_similarity"] < 1.0


# ---------------------------------------------------------------------------
# session_datetime
# ---------------------------------------------------------------------------

def test_session_datetime_derives_date_and_duration():
    rows = [
        {"id": 1, "text": "a", "start_time": 5.0, "end_time": 7.0,
         "timestamp": "2026-05-10 18:30:00"},
        {"id": 2, "text": "b", "start_time": 7.0, "end_time": 65.5,
         "timestamp": "2026-05-10 18:31:00"},
    ]
    info = ca.session_datetime(rows)
    assert info["date"] == "2026-05-10"
    assert info["start_wallclock"] == "18:30:00"
    assert info["duration_s"] == 60.5  # 65.5 - 5.0


def test_session_datetime_empty_rows():
    info = ca.session_datetime([])
    assert info["date"] is None and info["duration_s"] == 0.0


# ---------------------------------------------------------------------------
# pick_video
# ---------------------------------------------------------------------------

def _session(date, duration=3600.0):
    return {"date": date, "duration_s": duration}


def test_pick_video_exact_date():
    candidates = [
        {"id": "vA", "upload_date": "20260509", "title": "Sat", "duration": 3600},
        {"id": "vB", "upload_date": "20260510", "title": "Sun service", "duration": 3600},
        {"id": "vC", "upload_date": "20260517", "title": "Next Sun", "duration": 3600},
    ]
    result = ca.pick_video(candidates, _session("2026-05-10"))
    assert result["video_id"] == "vB"
    assert result["ambiguous"] is False


def test_pick_video_within_day_window():
    candidates = [{"id": "vA", "upload_date": "20260511", "title": "day after", "duration": 3600}]
    result = ca.pick_video(candidates, _session("2026-05-10"))
    assert result["video_id"] == "vA"  # +1 day still within window
    assert result["ranked"][0]["day_diff"] == 1


def test_pick_video_outside_window_excluded():
    candidates = [{"id": "vFar", "upload_date": "20260601", "title": "far", "duration": 3600}]
    result = ca.pick_video(candidates, _session("2026-05-10"))
    assert result["video_id"] is None
    assert result["ranked"] == []


def test_pick_video_ambiguous_same_day_ranks_by_duration():
    candidates = [
        {"id": "morning", "upload_date": "20260510", "title": "AM", "duration": 1800},
        {"id": "evening", "upload_date": "20260510", "title": "PM", "duration": 3600},
    ]
    result = ca.pick_video(candidates, _session("2026-05-10", duration=3550.0))
    assert result["ambiguous"] is True
    # Closest duration to the ~3550s session wins.
    assert result["video_id"] == "evening"


# --- title date parsing + title-driven resolution ---

def test_parse_title_datetime_month_and_time():
    dt = ca.parse_title_datetime("July 5, 2026 - 10:00 AM")
    assert dt is not None
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2026, 7, 5, 10, 0)
    pm = ca.parse_title_datetime("June 28, 2026 - 6:00 PM")
    assert (pm.month, pm.day, pm.hour) == (6, 28, 18)
    iso = ca.parse_title_datetime("Service 2026-07-05")
    assert (iso.year, iso.month, iso.day) == (2026, 7, 5)
    assert ca.parse_title_datetime("no date here") is None


def test_pick_video_prefers_title_date_over_approx_upload_date():
    # Reproduces the real bug: a July 5 session must NOT match June 28 videos whose
    # approximate upload_date rounded into the window, when a July 5 title exists.
    candidates = [
        {"id": "jun28am", "upload_date": "20260704", "title": "June 28, 2026 - 10:00 AM", "duration": 7212},
        {"id": "jul5am", "upload_date": "20260711", "title": "July 5, 2026 - 10:00 AM", "duration": 7813},
        {"id": "jul5pm", "upload_date": "20260711", "title": "July 5, 2026 - 6:00 PM", "duration": 5674},
    ]
    session = {"date": "2026-07-05", "duration_s": 7800.0, "start_wallclock": "09:32:18"}
    result = ca.pick_video(candidates, session, day_window=1)
    # Title date wins over the misleading approx upload_date, and 09:32 -> 10 AM service.
    assert result["video_id"] == "jul5am"
    assert result["ambiguous"] is True  # two July 5 services share the day


def test_pick_video_time_of_day_breaks_am_pm_tie():
    candidates = [
        {"id": "am", "upload_date": "20260705", "title": "July 5, 2026 - 10:00 AM", "duration": 4000},
        {"id": "pm", "upload_date": "20260705", "title": "July 5, 2026 - 6:00 PM", "duration": 4000},
    ]
    evening = {"date": "2026-07-05", "duration_s": 4000.0, "start_wallclock": "18:05:00"}
    assert ca.pick_video(candidates, evening, day_window=1)["video_id"] == "pm"
