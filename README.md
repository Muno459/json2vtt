# json2vtt / json2srt – Highly-configurable subtitle converter

`json2vtt.py` turns machine-transcribed JSON (ElevenLabs, Avanegar, Speechmatics, or a flat word list) into broadcast-grade WebVTT or SRT captions.  
It enforces hard timing/layout limits while offering smart heuristics for natural phrase breaks.

## Features

* **Multiple input schemas** – supports:
  1. ElevenLabs diarisation (`segments[] → words[]`).
  2. Generic `{"words": [...]}` arrays (e.g. Avanegar).
  3. Flat list of word objects – `[{"word": "hello", "start": 0.0, "end": 0.5}, …]`.
* **Quality guardrails** –
  * Max 7 s per cue / 2 lines / 42 chars per line.
  * CPS limit, dedupe window, speaker-change gap.
* **Natural phrase splitting** – soft flush on punctuation, standalone dashes, or long pauses.
* **Adaptive gap strategy** – default "shrink" avoids cumulative drift; legacy "shift" available.
* **Validation pass** – flags overlaps, gap violations, and timeline drift (strict mode aborts on failure).
* **Frame-accurate snap** – optional `--fps` to align to video frame boundaries.
* **Clean CLI** – every heuristic can be tuned or disabled via flags.

---
## Quick start

### 1. Convert JSON → VTT
```bash
python json2vtt.py my_transcript.json my_subs.vtt
```
Defaults: WebVTT, speaker prefixes off (change in code), smart gap handling, pause flush = 0.8 s.

### 2. Convert JSON → SRT
```bash
python json2vtt.py my_transcript.json my_subs.srt --srt
```

---
## CLI options (abridged)
```
Quality knobs:
  --dedupe-window SEC       Merge duplicate word within SEC (default 0.30)
  --min-cue SEC             Merge cues shorter than SEC (0 disables)
  --max-cps NUM             Characters-per-second limit (0 disables)
  --gap SEC                 Min gap at speaker change (default 0.08)
  --gap-strategy MODE       shrink|shift (default shrink)
  --pause-flush SEC         Flush cue after silence ≥ SEC (0 disables)

Style toggles:
  --no-software-flush       Disable early flush at punctuation
  --no-dash-flush           Ignore standalone dashes as boundaries
  --soft-flush-threshold F  Fraction of max-dur before commas trigger flush
  --no-speaker              Suppress ">> Speaker:" prefixes
  --srt                     Output SRT instead of VTT
  --fps FPS                 Frame-snap timestamps

Validation & logging:
  --strict                  Abort if validation fails
  --verbose                 Extra debug output

Run `python json2vtt.py --help` for the full list.
```

---
## Input schemas

1. **ElevenLabs diarisation**
```json5
{
  "segments": [
    {
      "speaker": {"name": "Alice"},
      "words": [{"text": "Hello", "start": 0.0, "end": 0.5}, ...]
    }
  ]
}
```
2. **Generic words array**
```json
{"words": [{"text": "Hello", "start": 0.0, "end": 0.5}]}
```
3. **Avanegar / Flat list**
```json
[{"word": "Hello", "start": 0.0, "end": 0.5}, {"word": "world", ...}]
```

---
## Examples

1. **Lower the pause threshold & snap to 25 fps**
```bash
python json2vtt.py talk.json talk.vtt --pause-flush 0.5 --fps 25
```

2. **Aggressive splitting (flush on every dash)**
```bash
python json2vtt.py doc.json out.srt --srt --dash-gap 0 --min-cue 0.4
```

3. **CI-safe run (abort on any issue)**
```bash
python json2vtt.py episode.json episode.vtt --strict --verbose
```

---
## FAQ

**Q: My cues drift late over time.**  
A: Use the default `--gap-strategy shrink` or reduce `--gap`.

**Q: I don't want dashes to break cues.**  
A: Add `--no-dash-flush`.

**Q: Can I keep speaker prefixes?**  
A: Uncomment `SPEAKER_PREFIX_FMT` in the source or pass `--no-speaker` to suppress.

**Q: Frame accuracy?**  
A: Use `--fps` with your video's frame rate; timestamps snap to frame boundaries.
---
## License
MIT 
