---
name: remake-song
description: Remake a song or melody with new synth sounds using this repo (midi-to-music). Use when the user gives an mp3/wav/MIDI and wants it redone with different sounds, wants the melody and bass notes pulled out of a song, wants to compare synth sounds on a melody, design a new sound, or build an arranged track or cue.
---

# Remake a song with midi-to-music

You cannot hear audio. The repo's checks tell you a render is in key, on time and clean; the user decides what
sounds good. Hand them files to play, describe each in one line, and ask what they like. Keep their choices.

Paths: `PY` = `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere. `music ...` = `music.cmd ...` on
Windows, `./music ...` elsewhere (both run `music.py`). Run everything from the repo root.

## 1. Setup (once; skip what exists)

```sh
uv venv .venv --python 3.12
uv pip install --python $PY -r requirements.txt
```
For audio input also: `uv pip install --python $PY --no-deps -r requirements-transcribe.txt`, and for a clean
melody/bass split `uv pip install --python $PY demucs torch torchaudio` (about 2 GB: tell the user first).

Optional, 3,000+ more sounds: Surge XT (free). Ask before downloading. Windows: the portable zip from
https://github.com/surge-synthesizer/releases-xt/releases unpacked to `~/tools/surge-xt`; macOS/Linux: the normal
installer. Presets with `"engine": "surge"` are skipped without it. Its first load takes about a minute.

## 2. Get the notes

**From a song:** trim to the part they want first (20-40 s). Renders loop the whole MIDI, so a full song is slow.
```sh
ffmpeg -ss 0:45 -t 30 -i song.mp3 clip.wav
$PY transcribe.py clip.wav                  # -> midi/clip.mid: tracks "melody" + "bass" (Basic Pitch on Demucs stems)
```
- Without Demucs it transcribes the full mix into one track, and the bass lands in "melody". Install Demucs for real use.
- The printed tempo is often wrong (seen: 110 detected for 83 BPM). Note times are exact in seconds, so plain
  renders are fine; for `--quantize` (snap to 16ths) or bar-based arrangements, ask the user the BPM and pass `--bpm`.
- A short repeating hook over a bass line transcribes cleaner with `--loop` (writes midi/melody.mid; check file
  midi/transcription_check.wav: song left, notes right).

**From a MIDI file:** it needs a track named `melody` (optional `bass`, and `bass_root` for held drone notes).
Rename tracks with mido if needed. **Write one:** follow `make_melodies.py`.

## 3. Audition sounds

```sh
music sounds --midi midi/clip.mid                      # every preset; or name a shortlist:
music sounds 14_broken 17_fm reese surge --midi midi/clip.mid
```
The user opens `renders/clip/index.html`: all sounds play in sync, loudness-matched, click or press 1-9 to switch.
Each preset's `"description"` in `presets/*.json` is your one-line pitch for it. On long clips render a shortlist
(each preset renders three passes of the MIDI).

## 4. Change or design a sound

Copy a preset in `presets/`, edit it, run `music sounds newname --midi ...`. Engines and effects are listed in
README.md; every parameter is in the docstrings of `engine.py` and `dsp/*.py`; `make_signature_presets.py` shows
deep chains. Note-aware effects (comb, sieve, formant, fshift, freeze) follow the lowest note, so use them on
single-note lines, not chords.

## 5. Build a track

Copy an arrangement in `arrangements/` (05_signature_rise is a good start), add `"midi": "midi/clip.mid"`, set
`"bars"`, swap layer presets, adjust `gain`/`lpf`/`hpf` keyframes `[[bar, value], ...]`, then `music arrange name`.
Output: `renders/arrangements/NAME.wav` (-14 LUFS), `NAME.mid`, `NAME_stems/`. Arrangements that use `bass_root`
are skipped for a MIDI without that track.

What worked before:
- Keep a bass under the melody from the first bar: the bass sets the key; a bare melody can sound like another key.
- Build with filter opening and layers entering; a short silence before the peak lands hard.
- Echo-heavy presets clutter syncopated melodies: lower their echo or swap to a drier preset at busy sections.
- End on the key's root chord, played by a preset without note-aware effects.

## 6. Check before handing over

```sh
music check [names] --midi midi/clip.mid     # every sound: notes on the grid, pitches heard = MIDI pitches
$PY qa.py                                     # loudness, true peak <= -1 dBTP, loop seam < 1, loudness per bar
```
`check` cannot hear notes under about 60 Hz, so it FAILs on bass notes in a melody track: that is the check's limit,
not a wrong note. Report the numbers in a line, then the files to play.

Remaking someone else's song is for personal use unless the user has the rights to it.
