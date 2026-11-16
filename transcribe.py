"""Audio to MIDI: writes a MIDI file music.py can render (music --midi out.mid sounds ...).

    python transcribe.py song.mp3                    # Basic Pitch (any rhythm, polyphonic) -> midi/song.mid
    python transcribe.py song.mp3 --no-split         # ... on the full mix: one "melody" track (default with Demucs: melody + bass)
    python transcribe.py song.mp3 --quantize         # snap onsets to a 16th grid; --bpm 90 overrides the detected tempo
    python transcribe.py song.mp3 --loop             # grid-fitted transcriber for a repeating hook (below), writes midi/melody.mid
    python transcribe.py --stems DIR [--loop]        # reuse stems already split (--loop needs other.wav and bass.wav)
    python transcribe.py song.mp3 --loop --bpm 90 --bars 8   # override the detected tempo or the loop length

Default: Spotify's Basic Pitch (uv pip install --no-deps -r requirements-transcribe.txt) on the Demucs stems: vocals+other+guitar+piano
-> track "melody", bass -> track "bass". Tempo from librosa beat tracking (or --bpm), tuning from the FFT peak of each note
(as below), both written to the MIDI. Notes are as heard: no grid unless --quantize. Quantized onsets sit on the 16th grid
through the first detected beat (MIDI time starts at that grid, less than one beat before the audio's start).

--loop method: Demucs (htdemucs_6s) splits the song; the melody is taken from the `other` stem, the harmony from `bass`.
The eighth-note grid is fitted by linear regression on the melody's note onsets (tempo from librosa as the first
guess). Each grid slot gets the pitch with the highest harmonic salience in the CQT; slots with no clear note become
rests. Bass: each bar's root, then per eighth whether it sits on the root or the octave above. The song's tuning
offset from A440 is measured per part and stored as text events `tuning_cents=...` that music.py applies.
Works best on a clear, repeating, eighth-note melody (a hook or ostinato), not on dense chords or free rhythm.
Writes a check file next to the MIDI: left = the song, right = the MIDI as sine tones.
"""
import argparse, importlib.util, os, subprocess, sys, tempfile
import numpy as np, librosa, mido, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
BPM, OFFSET, BARS = 120.0, 0.0, 4             # set by main() from the audio (or --bpm / --offset / --bars)
GATE = 0.875                                  # notes sound for ~88% of each eighth
TPB = 480                                     # MIDI ticks per quarter


def salience(y, sr, lo, hi, hop=256):
    """Harmonic-sum salience per MIDI pitch (rows lo..hi) over time; suppresses octave errors."""
    fmin = librosa.midi_to_hz(lo)
    n = (hi - lo) + 12 * 3 + 1
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=n, bins_per_octave=12))
    S = np.zeros((hi - lo + 1, C.shape[1]))
    for k in range(hi - lo + 1):
        for h, w in ((0, 1.0), (12, 0.5), (19, 0.33), (24, 0.25)):   # harmonics 1-4 in semitones
            if k + h < n:
                S[k] += w * C[k + h]
    return S, librosa.times_like(C, sr=sr, hop_length=hop)


def transcribe_slots(path, lo, hi, n_slots, slot_s, rest_ratio, win=(0.03, 0.25)):
    y, sr = librosa.load(path, sr=44100, mono=True)
    S, t = salience(y, sr, lo, hi)
    out = []
    for i in range(n_slots):
        t0 = OFFSET + i * slot_s
        m = (t >= t0 + win[0]) & (t < t0 + min(win[1], slot_s))
        s = S[:, m].mean(1)
        out.append((lo + int(s.argmax()), float(s.max())))
    peak = np.median([v for _, v in out])
    return [(p, v) if v >= rest_ratio * peak else (None, v) for p, v in out]


def octave_pattern(path, roots, eighth):
    """Per eighth: the bar's root, or root + 12 when the octave is clearly louder (the 808 bounces between them)."""
    y, sr = librosa.load(path, sr=44100, mono=True)
    lo = min(roots)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=256, fmin=librosa.midi_to_hz(lo), n_bins=max(roots) - lo + 13, bins_per_octave=12))
    t = librosa.times_like(C, sr=sr, hop_length=256)
    out = []
    for i in range(len(roots) * 8):
        r = roots[i // 8]; t0 = OFFSET + i * eighth
        m = (t >= t0 + 0.03) & (t < t0 + 0.25)
        root_e, oct_e = C[r - lo, m].mean(), C[r - lo + 12, m].mean()
        out.append(r + 12 if oct_e > 2 * root_e else r)          # octave at least 6 dB louder
    return out


def tuning_at(y, sr, events):
    """Median cents offset from equal temperament: FFT peak of each (start_s, midi) note, parabolic interpolation."""
    out = []
    for t0, p in events:
        seg = y[int((t0 + 0.03) * sr):int((t0 + 0.30) * sr)]
        N = 1 << 18
        S = np.log(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)) + 1e-12)
        f = librosa.midi_to_hz(p); k = int(f * N / sr)
        w = int(f * 0.03 * N / sr)                       # search +-3% around the note
        j = k - w + int(np.argmax(S[k - w:k + w]))
        d = 0.5 * (S[j - 1] - S[j + 1]) / (S[j - 1] - 2 * S[j] + S[j + 1])
        out.append(1200 * np.log2((j + d) * sr / N / f))
    return float(np.median(out))


def tuning(path, notes, eighth):
    y, sr = librosa.load(path, sr=44100, mono=True)
    return tuning_at(y, sr, [(OFFSET + i * eighth, p) for i, p in notes])


def grid(path, bpm=None, offset=None):
    """(bpm, first-note time): librosa's tempo as a guess, then a straight-line fit of the onsets to eighth notes."""
    y, sr = librosa.load(path, sr=44100, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False)
    if bpm is None:
        t = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
        e, o = 30 / t, on[0] if offset is None else offset
        for _ in range(6):                                          # refit: each pass pins more onsets to the right eighth
            idx = np.round((on - o) / e)
            keep = np.abs(on - (o + idx * e)) < 0.25 * e
            e, o = np.polyfit(idx[keep], on[keep], 1)
        bpm = 30 / e
        offset = o if offset is None else offset
    return bpm, (on[0] if offset is None else offset)


def split(audio):
    """Demucs htdemucs_6s stems of `audio` -> the folder holding them (separated/htdemucs_6s/<name>/)."""
    subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs_6s', '-o', os.path.join(HERE, 'separated'), audio], check=True)
    return os.path.join(HERE, 'separated', 'htdemucs_6s', os.path.splitext(os.path.basename(audio))[0])


def save_midi(out, bpm, end_tick, texts, tracks):
    """tracks: {name: [(start_tick, dur_tick, midi, vel)]}; texts: meta text events (tuning_cents=...)."""
    mid = mido.MidiFile(ticks_per_beat=TPB)
    meta = mido.MidiTrack(); mid.tracks.append(meta)
    meta += [mido.MetaMessage('track_name', name=os.path.basename(out), time=0),
             mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0),
             mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0)]
    meta += [mido.MetaMessage('text', text=t, time=0) for t in texts]
    meta.append(mido.MetaMessage('end_of_track', time=end_tick))
    for name, notes in tracks.items():
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage('track_name', name=name, time=0))
        ev = sorted([(s, 1, 'note_on', p, v) for s, d, p, v in notes] + [(s + d, 0, 'note_off', p, 0) for s, d, p, v in notes])
        last = 0
        for tick, _, kind, p, v in ev:
            tr.append(mido.Message(kind, note=p, velocity=v, time=tick - last, channel=list(tracks).index(name))); last = tick
        tr.append(mido.MetaMessage('end_of_track', time=end_tick - last))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mid.save(out)
    print('wrote', out)


def basic_pitch_notes(ys, sr, **kw):
    """Basic Pitch on each audio in ys -> [(start_s, dur_s, midi, amplitude 0-1)]; same-pitch notes never overlap (music.py pairs on/off by pitch)."""
    try: from basic_pitch.inference import predict
    except ImportError as e: sys.exit(f'{e}: uv pip install --no-deps -r requirements-transcribe.txt (or use --loop)')
    ev = []
    with tempfile.TemporaryDirectory() as d:
        for y in ys:
            if len(y) < sr // 4: continue                     # Basic Pitch raises on audio shorter than ~0.1 s
            f = os.path.join(d, 'in.wav'); sf.write(f, y, sr)
            ev += predict(f, **kw)[2]
    return no_overlap([(s, e - s, int(p), float(amp)) for s, e, p, amp, _ in ev])


def no_overlap(notes):
    """[(start, dur, midi, x)] with same-pitch notes cut at the next start (music.py pairs on/off by pitch); zero-length ones dropped."""
    out = []
    for s, d, p, x in sorted(notes, key=lambda n: (n[2], n[0])):
        if out and out[-1][2] == p and out[-1][0] + out[-1][1] > s: out[-1] = (out[-1][0], s - out[-1][0], p, out[-1][3])
        out.append((s, d, p, x))
    return sorted(n for n in out if n[1] > 0)


def basic(a):
    """Default mode: Basic Pitch on the stems (or the full mix), tempo from beat tracking, per-part tuning."""
    sr = 44100
    stem = a.stems or (None if a.no_split else split(a.audio) if importlib.util.find_spec('demucs') else None)
    if not stem and not a.no_split: print('Demucs not installed (pip install demucs): using the full mix, one "melody" track')
    if stem:
        wav = {n: librosa.load(os.path.join(stem, n + '.wav'), sr=sr, mono=True)[0] for n in ('vocals', 'other', 'guitar', 'piano', 'bass', 'drums')
               if os.path.exists(os.path.join(stem, n + '.wav'))}
        if not wav: sys.exit(f'no stems (vocals.wav, other.wav, bass.wav, ...) in {stem}')
        rms = {k: np.sqrt((v ** 2).mean()) for k, v in wav.items() if k not in ('bass', 'drums')}
        mel = [wav[k] for k in rms if rms[k] > 0.1 * max(rms.values())]      # stems within 20 dB of the loudest: quieter ones are separation bleed
        parts = {'melody': mel, 'bass': [wav['bass']]}
        mix = sum(wav.values())
    else:
        mix = librosa.load(a.audio, sr=sr, mono=True)[0]; parts = {'melody': [mix]}
    tempo, beats = librosa.beat.beat_track(y=mix, sr=sr, units='time')
    bpm = a.bpm or float(np.atleast_1d(tempo)[0])
    if not bpm > 0: bpm = 120.0; print('no beat found: tempo set to 120 BPM (use --bpm)')      # beat_track returns 0 on sparse or silent audio
    beat_s = 60 / bpm
    origin = float(beats[0]) % beat_s if a.quantize and len(beats) else 0.0      # 16th grid through the first beat
    q = TPB // 4 if a.quantize else 1
    tick = lambda t: max(0, int(round((t - origin) / beat_s * TPB / q)) * q)
    tracks, texts, end = {}, [], 0
    for name, ys in parts.items():
        y = sum(ys); notes = basic_pitch_notes(ys, sr, **(dict(minimum_frequency=30, maximum_frequency=500) if name == 'bass' else {}))
        long = [(s, p) for s, d, p, _ in notes if d >= 0.2]                     # short notes are too brief for an FFT peak
        c = tuning_at(y, sr, long) if long else 0.0
        texts.append(f'tuning_cents{"_bass" if name == "bass" else ""}={c:.1f}')
        tracks[name] = no_overlap([(tick(s), max(q, tick(s + d) - tick(s)), p, int(np.clip(round(127 * amp), 30, 127))) for s, d, p, amp in notes])   # snapping can overlap repeats
        end = max([end] + [s + d for s, d, _, _ in tracks[name]])
        print(f'{name}: {len(notes)} notes, tuning {c:+.1f} cents' + (f', {librosa.midi_to_note(min(n[2] for n in notes), unicode=False)}'
              f'..{librosa.midi_to_note(max(n[2] for n in notes), unicode=False)}' if notes else ''))
    if not tracks['melody']: sys.exit('no notes found')
    print(f'tempo: {bpm:.2f} BPM{" (--bpm)" if a.bpm else ""}{", quantized to 16ths" if a.quantize else ""}')
    save_midi(a.out, bpm, -(-end // (4 * TPB)) * 4 * TPB, texts, tracks)


def main():
    global BPM, OFFSET, BARS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio', nargs='?', help='song file (mp3/wav/...); split with Demucs into separated/')
    ap.add_argument('--stems', help='folder with Demucs stems (instead of audio); --loop needs other.wav and bass.wav')
    ap.add_argument('--bpm', type=float, help='tempo (default: detected)')
    ap.add_argument('--loop', action='store_true', help='grid-fitted hook transcriber instead of Basic Pitch')
    ap.add_argument('--no-split', action='store_true', help='Basic Pitch on the full mix only: one "melody" track')
    ap.add_argument('--quantize', action='store_true', help='Basic Pitch: snap note starts and lengths to a 16th grid')
    ap.add_argument('--offset', type=float, help='--loop: time of the first melody note in seconds (default: detected)')
    ap.add_argument('--bars', type=int, default=4, help='--loop: loop length in 4/4 bars (default 4)')
    ap.add_argument('--out', help='MIDI file to write (default: midi/<song>.mid, or midi/melody.mid with --loop)')
    a = ap.parse_args()
    if not (a.audio or a.stems): ap.error('give a song file or --stems')
    if a.bpm is not None and a.bpm <= 0: ap.error('--bpm must be positive')
    a.out = a.out or os.path.join(HERE, 'midi', 'melody.mid' if a.loop else os.path.splitext(os.path.basename((a.audio or a.stems).rstrip('/\\')))[0] + '.mid')
    if not a.loop: return basic(a)
    if a.audio and not a.stems: a.stems = split(a.audio)
    BARS = a.bars
    BPM, OFFSET = grid(os.path.join(a.stems, 'other.wav'), a.bpm, a.offset)
    print(f'grid: {BPM:.2f} BPM, first note at {OFFSET:.3f} s, {BARS} bars')
    eighth = 30.0 / BPM
    mel = transcribe_slots(os.path.join(a.stems, 'other.wav'), 55, 76, BARS * 8, eighth, 0.45)
    bass = transcribe_slots(os.path.join(a.stems, 'bass.wav'), 28, 52, BARS, eighth * 8, 0.2, win=(0.3, 2.5))
    vmax = max(v for p, v in mel if p)
    bass8 = octave_pattern(os.path.join(a.stems, 'bass.wav'), [p for p, _ in bass], eighth)
    cents = tuning(os.path.join(a.stems, 'other.wav'), [(i, p) for i, (p, _) in enumerate(mel) if p], eighth)
    cents_bass = tuning(os.path.join(a.stems, 'bass.wav'), list(enumerate(bass8)), eighth)   # the 808 is tuned differently
    print('melody:', ' '.join(librosa.midi_to_note(p, unicode=False) if p else '-' for p, _ in mel))
    print('bass:  ', ' '.join(librosa.midi_to_note(p, unicode=False) for p in bass8))
    print(f'tuning: melody {cents:+.1f} cents (A4 = {440 * 2 ** (cents / 1200):.1f} Hz), bass {cents_bass:+.1f} cents')

    e = TPB // 2
    tracks = {'melody': [(i * e, int(e * GATE), p, int(round(84 + 24 * (v / vmax) ** 0.7)))
                         for i, (p, v) in enumerate(mel) if p],
              'bass': [(i * e, e, p, 92 if i % 2 == 0 else 100) for i, p in enumerate(bass8)],   # octave a little hotter
              'bass_root': [(i * 8 * e, 8 * e, p, 96) for i, (p, _) in enumerate(bass)]}   # held roots, for drones
    save_midi(a.out, BPM, BARS * 4 * TPB, [f'tuning_cents={cents:.1f}', f'tuning_cents_bass={cents_bass:.1f}'], tracks)

    # Audible check: source mix with a soft sine rendering of the MIDI on top, aligned to the source grid.
    mix = a.audio or os.path.join(a.stems, 'other.wav')
    y, sr = librosa.load(mix, sr=44100, mono=True)
    tone = np.zeros_like(y)
    for s, d, p, v in tracks['melody']:
        t0 = OFFSET + s / TPB * 60 / BPM; n = int(d / TPB * 60 / BPM * sr); i0 = int(t0 * sr)
        tt = np.arange(n) / sr
        w = np.sin(2 * np.pi * librosa.midi_to_hz(p + cents / 100) * tt) * np.minimum(1, np.minimum(tt / 0.01, (n / sr - tt) / 0.03))
        tone[i0:i0 + n] += 0.25 * w[:len(tone) - i0]
    sf.write(os.path.join(os.path.dirname(a.out), 'transcription_check.wav'), np.stack([0.5 * y, tone], 1), sr, subtype='PCM_24')
    print('wrote transcription_check.wav (left: source, right: MIDI as sine)')


if __name__ == '__main__':
    main()
