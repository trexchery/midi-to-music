# midi-to-music

Music as code: write a melody, design sounds and build finished cues in Python, with automatic checks for key,
timing and loudness. Built so scripts and AI agents can make music without a DAW.

- **Everything is text.** Melody (MIDI), sounds (JSON presets) and arrangements (JSON) are files you can edit,
  diff and generate.
- **One melody, any sound.** Swap the instrument; notes, timing and key never change. 25 presets included.
- **It checks itself.** Every render is measured for key, note timing, pitch, loudness and clean loops.
- **Repeatable.** The same files always render the same audio.

Listen: [`demo/`](demo).

## Quick start

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python make_melodies.py                                   # demo melody -> midi/melody.mid
music                                                                   # Windows: music.cmd, else ./music
```

Open `renders/index.html` to A/B every sound in sync. Arrangements land in `renders/arrangements/`.

```
music sounds glass reese       only presets whose name contains "glass" or "reese"
music arrange fracture         only that arrangement
music check                    every sound plays the MIDI's notes on time and in tune
music --midi song.mid          any MIDI with a track named "melody" (optional: "bass", "bass_root")
python transcribe.py song.mp3  audio to MIDI with Basic Pitch (see below); --loop for a repeating hook
python qa.py                   loudness, true peak, loop seams, DC, stereo, loudness per bar
```

`transcribe.py` turns a song into `midi/song.mid`: [Basic Pitch](https://github.com/spotify/basic-pitch) (polyphonic, any
rhythm) on the melody and bass stems, with tempo and tuning written so `music --midi midi/song.mid` renders it in tune.
`--no-split` uses the full mix, `--quantize` snaps to a 16th grid, `--bpm` fixes a wrong tempo guess. Install:
`uv pip install --no-deps -r requirements-transcribe.txt` (plus `demucs torch torchaudio` to split). `--loop` is the
older transcriber for a clear, repeating melody over a bass line (Demucs, tempo grid fit, CQT): cleaner on that, useless elsewhere.

## Sounds

A preset (`presets/*.json`) is a synth `engine` plus an `fx` chain. Copy one, change it, run `music sounds name`.

- **Engines:** additive, FM, 4-operator FM stack, warped wavetable, phase distortion, modal (struck/bowed), subtractive,
  formant, granular, hard sync, Karplus-Strong, resampling (turns any preset into a wavetable), pitch-synchronous
  granular.
- **Effects:** any [pedalboard](https://github.com/spotify/pedalboard) effect; OTT-style compression, wavefolding,
  multiband distortion, clip stacks, lowpass gate, grid-locked filter sweeps; note-aware comb, vowel formant,
  harmonic sieve, frequency shifter and spectral freeze that re-tune to every note; generated impulse responses.
- **Optional:** the free [Valhalla Supermassive](https://valhalladsp.com/shop/reverb/valhalla-supermassive/) VST3
  (8 presets; falls back to a built-in reverb, set `SUPERMASSIVE_VST3` if it lives outside the standard folder).

Parameters are documented in the docstrings of `engine.py` and `dsp/*.py`; `python -m dsp.fm_warp` (and
`dsp.dynamics`, `dsp.noteaware`, `dsp.resample`) self-tests pitch, aliasing and determinism. Note-aware effects follow
the lowest note, so use them on melody lines, not chords.

## Arrangements

`arrangements/*.json`: layers of presets with `from`/`to` bars, `gain`/`lpf`/`hpf` keyframes, optional own `notes`,
`track` or `midi`, plus reversed swells. Each renders a WAV (-14 LUFS, -1 dBTP), a multitrack MIDI and stems.

## Limits

It measures, it does not listen: taste is yours. Offline rendering, not real-time. Tested on one style so far
(dark, cinematic, 4-bar ostinato).

## License

GPL-3.0 (pedalboard is GPL-3.0). The demo melody and presets are free to use in your music.
