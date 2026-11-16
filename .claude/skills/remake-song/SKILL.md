---
name: remake-song
description: Remake a song or melody with new synth sounds using this repo (midi-to-music). Use when the user gives an mp3/wav/MIDI and wants it redone with different sounds, wants the melody and bass notes pulled out of a song, wants to compare synth sounds on a melody, design a new sound, or build an arranged track or cue.
---

# Remake a song with midi-to-music

You cannot hear audio. The repo's checks tell you a render is in key, on time and clean; the user decides what
sounds good. Hand them files to play, describe each in one line, and ask what they like. Keep their choices.

Run everything from the repo root:
- `PY` = `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on macOS/Linux.
- `music` = `./music.cmd` on Windows (Git Bash; `music.cmd` in cmd/PowerShell), `./music` on macOS/Linux.
- `music sounds|arrange|check|list|player|all [names] [--midi file.mid]`; `music --help` for details. Never run
  `music` with no arguments: that is `all` and renders every sound and arrangement (10+ minutes).

## 1. Setup (once; skip what exists)

```sh
PY=.venv/Scripts/python.exe        # macOS/Linux: PY=.venv/bin/python
uv venv .venv --python 3.12
uv pip install --python $PY -r requirements.txt
uv pip install --python $PY --no-deps -r requirements-transcribe.txt     # only for audio input
```
Optional, ask first:
- Demucs (`uv pip install --python $PY demucs torch torchaudio`, about 2 GB): splits the song into stems, so the
  melody and bass come out cleaner. Without it the full mix is transcribed and notes under E3 become the bass.
- Surge XT (free synth, 3,000+ more sounds): Windows, the portable zip from
  https://github.com/surge-synthesizer/releases-xt/releases unpacked to `~/tools/surge-xt`; macOS/Linux, the
  normal installer. Presets with `"engine": "surge"` are skipped without it; first load takes about a minute.

## 2. Get the notes

**From a song.** Check its length (`ffprobe -v error -show_entries format=duration -of csv=p=0 song.mp3`). Over
40 s, ask which part and trim to 20-40 s: every render loops the whole MIDI.
```sh
ffmpeg -ss 0:45 -t 30 -i song.mp3 clip.wav
$PY transcribe.py clip.wav                  # -> midi/clip.mid: tracks "melody" and "bass"
```
**Always confirm the tempo before rendering.** Detection is often wrong (seen: 110 and 167 for an 83 BPM song).
Ask the user for the BPM (or tap it out), then rerun with `--bpm N`. A wrong tempo puts the loop and every
arrangement bar in the wrong place. Other flags: `--quantize` snaps to 16ths; `--bass-below N` moves the
melody/bass split (full mix only); `--loop` is cleaner for a short repeating hook over a bass line.

**From a MIDI file:** it needs a track named `melody` (optional: `bass`; `bass_root` = held notes for drones).
Rename tracks with mido if needed. **Write one:** follow `make_melodies.py`.

## 3. Audition sounds

`music list` prints every preset with the track it plays and a one-line description (use it to pick and pitch).
Render a shortlist of 3-5, never all at once:
```sh
music sounds 08_reese 14_broken 22_vocal --midi midi/clip.mid
```
The user opens `renders/clip/index.html`: all sounds in sync, loudness-matched, with the bass under them; click or
press 1-9 to switch. Timing: most presets render in 5-20 s per 10 s of music; 18_steel_resample, 22_vocal_reese,
23_grain_monolith and Surge are slower (30-60 s), so use a 10-minute timeout or run in the background.
Starting points: dark/electronic 14_broken_signal, 17_fm_snarl, 22_vocal_reese, 20_phase_zap, 08_reese_menace;
glassy 19_glass_from_grit, 01_glass_harmonica; warm 07_analog_warmth, 03_dark_tines. 15_sub_drone and 16_808_bass
are bass/drone layers, not leads.

## 4. Change or design a sound

Copy a preset in `presets/`, edit it, run `music sounds newname --midi ...`. Engines and effects are listed in
README.md; every parameter is in the docstrings of `engine.py` and `dsp/*.py`; `make_signature_presets.py` shows
deep chains. Note-aware effects (comb, sieve, formant, fshift, freeze) follow the lowest note: single lines only.

## 5. Build a track

Write `arrangements/NAME.json`, then `music arrange NAME`. A short build-up (bars are 0-based, `to` is exclusive):
```json
{"description": "build-up", "midi": "midi/clip.mid", "bars": 8, "tail": 6, "lufs": -14,
 "layers": [
  {"preset": "16_808_bass", "from": 0, "to": 8, "gain": [[0, -8], [6.9, 0], [7, -40], [7.99, -40], [8, 2]]},
  {"preset": "08_reese_menace", "from": 0, "to": 8, "gain": [[0, -10], [6.9, -2], [7, -40], [7.99, -40]], "lpf": [[0, 500], [7, 9000]]},
  {"preset": "25_broken_signal_tight", "from": 4, "to": 8, "gain": [[4, -14], [6.9, -4], [7, -40]]},
  {"preset": "07_analog_warmth", "from": 8, "to": 9, "notes": [[0, 8, 65, 100], [0, 8, 68, 104], [0, 8, 72, 112]], "gain": [[8, -2]], "tail": 6},
  {"preset": "16_808_bass", "from": 8, "to": 9, "notes": [[0, 8, 29, 112]], "gain": [[8, 1]], "tail": 6}
 ],
 "swells": [{"preset": "18_steel_resample", "bar": 8, "beats": 4, "gain": -4}]}
```
- `gain` is dB keyframes `[[bar, dB]]` (0 dB = the same loudness for every preset), `lpf`/`hpf` are Hz keyframes.
  Layers play the MIDI's melody unless they set `"track"` or inline `"notes"` `[beat, beats, midi, velocity]`.
- The final chord above is F minor (F4 Ab4 C5 over F1). Transpose it to the song's key: find the key from the
  MIDI's most common pitch classes and its last bass note.
- Bass layers need a `bass` track; layers using `bass_root` are skipped when the MIDI has none.
- Keep a bass under the melody from the first bar: the bass sets the key; a bare melody can sound like another key.
- A short silence before the peak lands hard; echo-heavy presets clutter syncopated melodies (use the `_tight` one).
Output: `renders/arrangements/NAME.wav` (-14 LUFS), `NAME.mid`, `NAME_stems/`.

## 6. Check before handing over

```sh
music check 08_reese 14_broken --midi midi/clip.mid   # notes on the grid, pitches heard = MIDI pitches
$PY qa.py                                             # every file in renders/: loudness, true peak, seams, per bar
```
Pass: check says `ok`; loops -16 LUFS, arrangements -14 +/- 0.5 LUFS, true peak at or under -1.0 dBTP, seam under
1; the per-bar column should rise through a build. `check` cannot hear notes under about 60 Hz, so a FAIL on bass
notes inside a melody track is the check's limit, not a wrong note. Report the numbers in one line, then the
files to play.

Remaking someone else's song is for personal use unless the user has the rights to it.
