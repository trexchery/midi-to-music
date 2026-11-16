"""Audio to MIDI for a repeating melody over a bass line: writes a MIDI file music.py can render.

    python transcribe.py song.mp3                    # splits it with Demucs, finds tempo + grid, writes midi/melody.mid
    python transcribe.py --stems DIR                 # reuse stems already split (needs other.wav and bass.wav)
    python transcribe.py song.mp3 --bpm 90 --bars 8  # override the detected tempo or the loop length

Method: Demucs (htdemucs_6s) splits the song; the melody is taken from the `other` stem, the harmony from `bass`.
The eighth-note grid is fitted by linear regression on the melody's note onsets (tempo from librosa as the first
guess). Each grid slot gets the pitch with the highest harmonic salience in the CQT; slots with no clear note become
rests. Bass: each bar's root, then per eighth whether it sits on the root or the octave above. The song's tuning
offset from A440 is measured per part and stored as text events `tuning_cents=...` that music.py applies.
Works best on a clear, repeating, eighth-note melody (a hook or ostinato), not on dense chords or free rhythm.
Writes a check file next to the MIDI: left = the song, right = the MIDI as sine tones.
"""
import argparse, os
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


def tuning(path, notes, eighth):
    """Median cents offset from equal temperament: FFT peak of each transcribed note, parabolic interpolation."""
    y, sr = librosa.load(path, sr=44100, mono=True)
    out = []
    for i, p in notes:
        seg = y[int((OFFSET + i * eighth + 0.03) * sr):int((OFFSET + i * eighth + 0.30) * sr)]
        N = 1 << 18
        S = np.log(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)) + 1e-12)
        f = librosa.midi_to_hz(p); k = int(f * N / sr)
        w = int(f * 0.03 * N / sr)                       # search +-3% around the note
        j = k - w + int(np.argmax(S[k - w:k + w]))
        d = 0.5 * (S[j - 1] - S[j + 1]) / (S[j - 1] - 2 * S[j] + S[j + 1])
        out.append(1200 * np.log2((j + d) * sr / N / f))
    return float(np.median(out))


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


def main():
    global BPM, OFFSET, BARS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio', nargs='?', help='song file (mp3/wav/...); split with Demucs into separated/')
    ap.add_argument('--stems', help='folder with Demucs stems other.wav and bass.wav (instead of audio)')
    ap.add_argument('--bpm', type=float, help='tempo (default: detected)')
    ap.add_argument('--offset', type=float, help='time of the first melody note in seconds (default: detected)')
    ap.add_argument('--bars', type=int, default=4, help='loop length in 4/4 bars (default 4)')
    ap.add_argument('--out', default=os.path.join(HERE, 'midi', 'melody.mid'))
    a = ap.parse_args()
    if not (a.audio or a.stems): ap.error('give a song file or --stems')
    if a.audio and not a.stems:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs_6s', '-o', os.path.join(HERE, 'separated'), a.audio], check=True)
        a.stems = os.path.join(HERE, 'separated', 'htdemucs_6s', os.path.splitext(os.path.basename(a.audio))[0])
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

    mid = mido.MidiFile(ticks_per_beat=TPB)
    meta = mido.MidiTrack(); mid.tracks.append(meta)
    meta += [mido.MetaMessage('track_name', name=os.path.basename(a.out), time=0),
             mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0),
             mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0),
             mido.MetaMessage('text', text=f'tuning_cents={cents:.1f}', time=0),
             mido.MetaMessage('text', text=f'tuning_cents_bass={cents_bass:.1f}', time=0),
             mido.MetaMessage('end_of_track', time=BARS * 4 * TPB)]
    e = TPB // 2
    tracks = {'melody': [(i * e, int(e * GATE), p, int(round(84 + 24 * (v / vmax) ** 0.7)))
                         for i, (p, v) in enumerate(mel) if p],
              'bass': [(i * e, e, p, 92 if i % 2 == 0 else 100) for i, p in enumerate(bass8)],   # octave a little hotter
              'bass_root': [(i * 8 * e, 8 * e, p, 96) for i, (p, _) in enumerate(bass)]}   # held roots, for drones
    for name, notes in tracks.items():
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage('track_name', name=name, time=0))
        ev = sorted([(s, 1, 'note_on', p, v) for s, d, p, v in notes] + [(s + d, 0, 'note_off', p, 0) for s, d, p, v in notes])
        last = 0
        for tick, _, kind, p, v in ev:
            tr.append(mido.Message(kind, note=p, velocity=v, time=tick - last, channel=list(tracks).index(name))); last = tick
        tr.append(mido.MetaMessage('end_of_track', time=BARS * 4 * TPB - last))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    mid.save(a.out)
    print('wrote', a.out)

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
