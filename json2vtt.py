#!/usr/bin/env python3
import argparse, json, sys, textwrap, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --------------------------------------------------------------------------
# Hard limits
# --------------------------------------------------------------------------
MAX_SEC_PER_CUE   = 7.0
MAX_LINES_PER_CUE = 2
MAX_CHARS_PER_LINE = 42
DEFAULT_MIN_CUE_SEC = 0.6          # merge cues shorter than this
DEFAULT_DEDUPE_WINDOW = 0.30       # 'یعنی یعنی‌…' filter (s)
DEFAULT_SPEAKER_GAP = 0.08         # force small gap on speaker change (s)
DEFAULT_PUNCT_CHARS = set("—–،.؟?!…؛:") | {",", "."}  # include dashes & LTR marks
SENTENCE_END_CHARS = set(".؟?!…؛:")  # strong punctuation that naturally ends a phrase
DEFAULT_CPS_LIMIT = 0              # 0 = off
DEFAULT_FPS = 0                    # 0 = do not frame-snap
DEFAULT_SOFT_FLUSH_THRESHOLD = 0.6  # fraction of MAX_SEC_PER_CUE before soft flush kicks in
DEFAULT_DASH_THRESHOLD = 0.4         # min silence (s) after dash to trigger flush
MAX_TOTAL_DRIFT_WARN = 0.5            # warn if total drift exceeds this (s)
DEFAULT_PAUSE_FLUSH = 0.8             # flush cue if silence ≥ this (s) between words (0=off)
#SPEAKER_PREFIX_FMT = ">> {speaker}: "  # Prefix added on speaker change
SPEAKER_PREFIX_FMT = ""  # Prefix added on speaker change
DASH_BOUNDARY_TOKENS = {"--", "—", "–"}  # stand-alone tokens treated as strong pauses

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def dbg(msg: str, args) -> None:
    if args.verbose:
        print(f"[DEBUG] {msg}", file=sys.stderr)

def seconds_to_timestamp(sec: float, millis_sep=".", fps: int = 0, is_end=False) -> str:
    """
    Return HH:MM:SS.mmm (WebVTT) or HH:MM:SS,mmm (SRT) with optional frame snapping.
    If fps > 0, snap start down (floor) and end up (ceil) to frame boundaries.
    """
    if fps:
        frame_len = 1 / fps
        sec = (int(sec / frame_len) + (1 if is_end else 0)) * frame_len
    hours, remainder = divmod(int(sec), 3600)
    minutes, seconds = divmod(remainder, 60)
    millis = int(round((sec - int(sec)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{millis_sep}{millis:03d}"

def wrap_ok(text: str) -> bool:
    wrapped = textwrap.wrap(text, width=MAX_CHARS_PER_LINE,
                            break_long_words=False, break_on_hyphens=False)
    return len(wrapped) <= MAX_LINES_PER_CUE

# --------------------------------------------------------------------------
# Word ingest & cleaning
# --------------------------------------------------------------------------
def load_words(data: Dict[str, Any],
               glue_punct: bool,
               dedupe_window: float,
               punct_chars=set(DEFAULT_PUNCT_CHARS),
               args=None) -> List[Dict[str, Any]]:
    """
    Flatten JSON → list of word dicts with optional punctuation glue and stutter removal.
    """
    words: List[Dict[str, Any]] = []
    prev_word: Optional[Dict[str, Any]] = None

    def process_word(w: Dict[str, Any], speaker_name: str = "") -> None:
        """Common logic to ingest a single word token regardless of schema."""
        nonlocal prev_word

        txt = w.get("text") or w.get("word", "")
        if not txt.strip():
            return

        # Skip explicit spacing tokens
        if w.get("type") == "spacing":
            return

        # Attach standalone punctuation to previous token
        if glue_punct and txt.strip() in punct_chars and prev_word:
            prev_word["text"] += txt
            prev_word["end"] = w.get("end_time", w.get("end"))
            return

        # Normalise timing field names
        start_t = w.get("start_time", w.get("start"))
        end_t   = w.get("end_time",   w.get("end"))
        if start_t is None or end_t is None:
            return

        start_t = float(start_t)
        end_t   = float(end_t)

        # Remove near-duplicate stutters within dedupe_window
        if (dedupe_window and prev_word and txt == prev_word["text"]
                and start_t - prev_word["start"] <= dedupe_window):
            prev_word["end"] = end_t
            return

        word_dict = dict(text=txt.strip(),
                         start=start_t,
                         end=end_t,
                         speaker=speaker_name)
        words.append(word_dict)
        prev_word = word_dict

    # ------------------------------------------------------------------
    # Schema C: raw list of word dicts e.g. [{"word":..., "start":..., "end":...}, ...]
    # ------------------------------------------------------------------
    if isinstance(data, list):
        for w in data:
            process_word(w, "")

    # ------------------------------------------------------------------
    # Schema A: ElevenLabs diarization → segments/words hierarchy
    # ------------------------------------------------------------------
    elif isinstance(data, dict) and data.get("segments"):
        for seg in data.get("segments", []):
            speaker_name = seg.get("speaker", {}).get("name", "")
            for w in seg.get("words", []):
                process_word(w, speaker_name)

    # ------------------------------------------------------------------
    # Schema B: Top-level "words" array (e.g. Whisper or other services)
    # ------------------------------------------------------------------
    elif isinstance(data, dict) and data.get("words"):
        for w in data["words"]:
            speaker = w.get("speaker_id") or w.get("speaker", "")
            process_word(w, speaker)

    dbg(f"Loaded {len(words)} words", args)
    return words

# --------------------------------------------------------------------------
# Cue builder
# --------------------------------------------------------------------------
def build_cues(words: List[Dict[str, Any]],
               speaker_prefix_fmt: str,
               min_cue_sec: float,
               cps_limit: float,
               speaker_gap: float,
               flush_early: bool,
               soft_flush_threshold: float,
               flush_dashes: bool,
               dash_threshold: float,
               boundary_chars: set = SENTENCE_END_CHARS,
               gap_strategy: str = "shrink",
               pause_flush: float = DEFAULT_PAUSE_FLUSH,
               args=None) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []
    buf_words: List[str] = []
    buf_start: Optional[float] = None
    buf_end: Optional[float] = None
    buf_speaker: str = ""
    dash_pending: bool = False      # flagged when we just added a dash token
    dash_end_time: float = 0.0
    last_speaker: str = ""

    def flush_buffer(force=False):
        nonlocal buf_words, buf_start, buf_end, buf_speaker, last_speaker
        if not buf_words:
            return
        text = " ".join(buf_words).strip()
        if buf_speaker and buf_speaker != last_speaker and speaker_prefix_fmt:
            text = speaker_prefix_fmt.format(speaker=buf_speaker) + text
        cues.append((buf_start, buf_end, text))
        last_speaker = buf_speaker
        buf_words.clear()
        buf_start = buf_end = None
        buf_speaker = ""

    for w in words:
        if not buf_words:
            buf_words = [w["text"]]
            buf_start = w["start"]
            buf_end = w["end"]
            buf_speaker = w["speaker"]
            continue

        candidate_text = " ".join(buf_words + [w["text"]]).strip()
        candidate_dur = w["end"] - buf_start

        near_time_limit = candidate_dur >= soft_flush_threshold * MAX_SEC_PER_CUE

        speaker_change = (w["speaker"] != buf_speaker)

        # Flush if long pause separates previous word and current word
        if (pause_flush > 0 and buf_words and w["start"] - (buf_end or 0) >= pause_flush):
            flush_buffer()
            # treat as new buffer starting with this word
            buf_words = [w["text"]]
            buf_start = w["start"]
            buf_end = w["end"]
            buf_speaker = w["speaker"]
            # We intentionally keep processing with newly initialised buffer (skip rest checks)
            continue

        # If previous token was a dash and the gap/speaker change warrants flush
        if dash_pending and (
                (w["start"] - dash_end_time >= dash_threshold) or
                (w["speaker"] != buf_speaker)):
            flush_buffer()
            # start new buffer with this word
            buf_words = [w["text"]]
            buf_start = w["start"]
            buf_end = w["end"]
            buf_speaker = w["speaker"]
            dash_pending = False
            continue

        dash_pending = False  # reset unless set later

        # Check limits
        time_violate = candidate_dur > MAX_SEC_PER_CUE
        wrap_violate = not wrap_ok(
            (speaker_prefix_fmt.format(speaker=buf_speaker)
             if buf_speaker != last_speaker and speaker_prefix_fmt else "") + candidate_text)
        cps_violate = (cps_limit and candidate_dur > 0
                       and len(candidate_text) / candidate_dur > cps_limit)

        if time_violate or wrap_violate or cps_violate:
            flush_buffer()
            buf_words = [w["text"]]
            buf_start = w["start"]
            buf_end = w["end"]
            buf_speaker = w["speaker"]
            continue

        buf_words.append(w["text"])
        buf_end = w["end"]

        # Handle standalone dash token logic AFTER adding it, so it stays with current cue
        if flush_dashes and w["text"] in DASH_BOUNDARY_TOKENS:
            dash_pending = True
            dash_end_time = w["end"]

        # Optional heuristic: if the current word ends with strong punctuation and
        # the cue is already long enough, flush early so the cue ends at a
        # natural linguistic boundary. When the cue is nearing the hard
        # duration limit, accept slightly weaker punctuation (commas, semicolons).
        weak_boundary_chars = {",", "،", ";"}
        dash_boundary_token = w["text"] in DASH_BOUNDARY_TOKENS
        effective_boundary = boundary_chars | (weak_boundary_chars if near_time_limit else set())

        if (flush_early and buf_words and buf_end - buf_start >= min_cue_sec
                and (dash_boundary_token or (w["text"] and w["text"][-1] in effective_boundary))):
            flush_buffer()
            continue

    flush_buffer()

    # Pass 2 – merge cues shorter than min_cue_sec with neighbours where safe
    if min_cue_sec > 0:
        merged: List[Tuple[float, float, str]] = []
        for cue in cues:
            if (merged and cue[1] - cue[0] < min_cue_sec):
                prev = merged[-1]
                # Don't merge across strong sentence boundary
                if prev[2].strip().endswith(tuple(boundary_chars)):
                    merged.append(cue)
                    continue

                merged_text = prev[2] + " " + cue[2]
                if (cue[1] - prev[0] <= MAX_SEC_PER_CUE) and wrap_ok(merged_text):
                    merged[-1] = (prev[0], cue[1], merged_text)
                    continue
            merged.append(cue)
        cues = merged

    # Pass 3 – enforce minimal gap between cues without drifting timeline
    if speaker_gap > 0 and len(cues) > 1:
        for i in range(1, len(cues)):
            prev_start, prev_end, prev_text = cues[i-1]
            start, end, text = cues[i]
            gap = start - prev_end

            if gap >= speaker_gap:
                continue  # already enough space

            if gap_strategy == "shift":
                # Old behaviour: push current cue later (may drift timeline)
                shift = speaker_gap - gap
                start += shift
                end = min(end + shift, start + MAX_SEC_PER_CUE)
                cues[i] = (start, end, text)
            else:  # "shrink" (default): shorten previous cue end time
                new_prev_end = max(prev_start, start - speaker_gap)
                if new_prev_end <= prev_start:  # can't shrink, fallback to shift minimal amount
                    shift = speaker_gap - gap
                    start += shift
                    end = min(end + shift, start + MAX_SEC_PER_CUE)
                    cues[i] = (start, end, text)
                else:
                    cues[i-1] = (prev_start, new_prev_end, prev_text)

    dbg(f"Built {len(cues)} cues", args)
    return cues

# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def write_vtt(cues, fh, fps, args):
    print("WEBVTT\n", file=fh)
    for idx, (start, end, text) in enumerate(cues, 1):
        print(idx, file=fh)
        s = seconds_to_timestamp(start, ".", fps)
        e = seconds_to_timestamp(end,   ".", fps, is_end=True)
        print(f"{s} --> {e}", file=fh)
        for line in textwrap.wrap(text, MAX_CHARS_PER_LINE,
                                  break_long_words=False, break_on_hyphens=False)[:MAX_LINES_PER_CUE]:
            print(line, file=fh)
        print(file=fh)

def write_srt(cues, fh, fps, args):
    for idx, (start, end, text) in enumerate(cues, 1):
        print(idx, file=fh)
        s = seconds_to_timestamp(start, ",", fps)
        e = seconds_to_timestamp(end,   ",", fps, is_end=True)
        print(f"{s} --> {e}", file=fh)
        for line in textwrap.wrap(text, MAX_CHARS_PER_LINE,
                                  break_long_words=False, break_on_hyphens=False)[:MAX_LINES_PER_CUE]:
            print(line, file=fh)
        print(file=fh)

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_cues(cues: List[Tuple[float, float, str]],
                  last_word_end: float,
                  speaker_gap: float,
                  strict: bool,
                  args=None) -> None:
    """Basic sanity checks – overlap, gap, drift."""
    issues = 0
    for i in range(1, len(cues)):
        prev_start, prev_end, _ = cues[i-1]
        cur_start, cur_end, _ = cues[i]
        if cur_start < prev_end:
            print(f"⚠️  Overlap: cue {i} starts before previous ends", file=sys.stderr)
            issues += 1
        if cur_start - prev_end < speaker_gap - 1e-3:  # allow tiny float err
            print(f"⚠️  Gap < speaker_gap ({cur_start - prev_end:.3f}s)", file=sys.stderr)
            issues += 1

    drift = abs(cues[-1][1] - last_word_end)
    if drift > MAX_TOTAL_DRIFT_WARN:
        print(f"⚠️  Timeline drift {drift:.2f}s compared to last word", file=sys.stderr)
        issues += 1

    if strict and issues:
        sys.exit(f"❌ Validation failed with {issues} issue(s). Use --no-strict to override.")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="ElevenLabs diarization JSON → broadcast‑grade WebVTT / SRT")
    ap.add_argument("json_file", type=Path,
                    help="Input ElevenLabs JSON file")
    ap.add_argument("out_file", nargs="?",
                    type=Path, help="Output subtitle file (default: stdout)")
    # Quality knobs
    ap.add_argument("--keep-spaces-around-punct", action="store_true",
                    help="Do NOT glue standalone punctuation to previous word")
    ap.add_argument("--dedupe-window", type=float, default=DEFAULT_DEDUPE_WINDOW,
                    help="Skip duplicate word if repeats within this many seconds (0 = off)")
    ap.add_argument("--min-cue", type=float, default=DEFAULT_MIN_CUE_SEC,
                    help="Merge cues shorter than this many seconds (0 = off)")
    ap.add_argument("--max-cps", type=float, default=DEFAULT_CPS_LIMIT,
                    help="Max characters/sec before splitting (0 = off)")
    ap.add_argument("--gap", type=float, default=DEFAULT_SPEAKER_GAP,
                    help="Minimal gap (s) at speaker change (0 = off)")
    ap.add_argument("--gap-strategy", choices=["shrink", "shift"], default="shrink",
                    help="How to enforce gap on speaker change: 'shrink' previous cue (default) or 'shift' next cue (legacy)")
    ap.add_argument("--no-soft-phrase-flush", action="store_true",
                    help="Do NOT flush cues early at strong sentence-ending punctuation")
    ap.add_argument("--soft-flush-threshold", type=float,
                    default=DEFAULT_SOFT_FLUSH_THRESHOLD,
                    help="Fraction of max cue length before weaker punctuation (comma, semicolon) triggers a soft flush; 0-1 range")
    # Style toggles
    ap.add_argument("--no-speaker", action="store_true",
                    help='Suppress ">> Speaker X:" prefixes')
    ap.add_argument("--srt", action="store_true",
                    help="Write SRT instead of WebVTT")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help="Snap timestamps to given frame rate (0 = no snapping)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print debugging info to stderr")
    # Dash handling
    ap.add_argument("--no-dash-flush", action="store_true",
                    help="Do NOT end a cue after a stand-alone dash token")
    ap.add_argument("--dash-gap", type=float, default=DEFAULT_DASH_THRESHOLD,
                    help="Silence (s) after stand-alone dash required to flush cue (0 = always flush)")
    ap.add_argument("--pause-flush", type=float, default=DEFAULT_PAUSE_FLUSH,
                    help="Flush cue if silence between words ≥ this many seconds (0=disabled)")
    ap.add_argument("--strict", action="store_true",
                    help="Abort if validation detects overlaps, gap violations or large drift")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    try:
        data = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    except Exception as exc:
        sys.exit(f"❌ Failed to read/parse JSON: {exc}")

    words = load_words(
        data,
        glue_punct=not args.keep_spaces_around_punct,
        dedupe_window=args.dedupe_window,
        args=args)

    if not words:
        sys.exit("❌ No usable words in JSON.")

    speaker_prefix = "" if args.no_speaker else SPEAKER_PREFIX_FMT
    cues = build_cues(
        words,
        speaker_prefix_fmt=speaker_prefix,
        min_cue_sec=args.min_cue,
        cps_limit=args.max_cps,
        speaker_gap=args.gap,
        flush_early=not args.no_soft_phrase_flush,
        soft_flush_threshold=max(0.0, min(1.0, args.soft_flush_threshold)),
        flush_dashes=not args.no_dash_flush,
        dash_threshold=max(0.0, args.dash_gap),
        boundary_chars=SENTENCE_END_CHARS,
        gap_strategy=args.gap_strategy,
        pause_flush=max(0.0, args.pause_flush),
        args=args)

    # Validation pass
    validate_cues(cues, last_word_end=words[-1]['end'],
                  speaker_gap=args.gap, strict=args.strict, args=args)

    out_fh = (args.out_file.open("w", encoding="utf-8")
              if args.out_file else sys.stdout)

    if args.srt:
        write_srt(cues, out_fh, fps=args.fps, args=args)
    else:
        write_vtt(cues, out_fh, fps=args.fps, args=args)

    if args.out_file:
        out_fh.close()
        dbg(f"Wrote {args.out_file}", args)

if __name__ == "__main__":
    main()