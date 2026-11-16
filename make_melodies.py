"""Writes the demo melodies to midi/ (F minor, 83.13 BPM, 4-bar loops). They are only examples: any MIDI file with a
`melody` track (and optionally `bass` and `bass_root`) works, from a DAW or from transcribe.py.

    python make_melodies.py                   # writes midi/melody.mid (the default) and midi/melody_alt.mid
    music --midi midi/melody_alt.mid          # render with the other one (output goes to renders/melody_alt/)

The tracks: melody, bass (808 pattern: root / octave on every eighth), bass_root (held roots, for drones).
"""
import os, mido
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
BPM, TPB, BARS = 83.13, 480, 4
M = lambda n: int(librosa.note_to_midi(n))


def bounce(roots):
    """Root / octave on every eighth."""
    return [(b * 4 + i * 0.5, 0.5, M(r) + 12 * (i % 2), 92 if i % 2 == 0 else 100) for b, r in enumerate(roots) for i in range(8)]


def lock(ending):
    """A 5-note cell (F C Ab Db C) on a 3+3+2 sixteenth rhythm: 6 onsets per bar against 5 notes, so the accent lands
    on a different note every bar (a polymeter). `ending` replaces the last two notes of bar 4."""
    cell, onsets, lens = ['F4', 'C5', 'Ab4', 'Db5', 'C5'], [0, 3, 6, 8, 11, 14], [3, 3, 2, 3, 3, 2]
    mel, i = [], 0
    for bar in range(BARS):
        for k, (o, l) in enumerate(zip(onsets, lens)):
            n = ending[k - 4] if bar == 3 and k >= 4 else cell[i % 5]
            mel.append((bar * 4 + o / 4, l / 4 * 0.85, M(n), 112 if k in (0, 3) else 92 + 4 * (k % 2)))
            i += 1
    roots = ['F1', 'Db2', 'Bb1', 'C2']
    return mel, bounce(roots), roots


def save(path, mel, bass, roots):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    end = BARS * 4 * TPB
    mid.tracks.append(mido.MidiTrack([
        mido.MetaMessage('track_name', name=os.path.basename(path), time=0),
        mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0),
        mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0),
        mido.MetaMessage('end_of_track', time=end)]))
    tracks = {'melody': mel, 'bass': bass, 'bass_root': [(b * 4, 4, M(r), 96) for b, r in enumerate(roots)]}
    for ch, (name, notes) in enumerate(tracks.items()):
        tr = mido.MidiTrack([mido.MetaMessage('track_name', name=name, time=0)]); mid.tracks.append(tr)
        ev = sorted([(round(s * TPB), 1, 'note_on', p, v) for s, d, p, v in notes] +
                    [(round((s + d) * TPB), 0, 'note_off', p, 0) for s, d, p, v in notes])
        assert ev[-1][0] <= end, f'{name} runs past the loop'
        last = 0
        for tick, _, kind, p, v in ev:
            tr.append(mido.Message(kind, note=p, velocity=v, time=tick - last, channel=ch)); last = tick
        tr.append(mido.MetaMessage('end_of_track', time=end - last))
    mid.save(path)


if __name__ == '__main__':
    for name, ending in [('melody', ('Eb5', 'C5')),        # reaches up, then C falls to the loop's first F
                         ('melody_alt', ('Ab4', 'G4'))]:   # steps down Ab-G into the loop's first F
        mel, bass, roots = lock(ending)
        save(os.path.join(HERE, 'midi', f'{name}.mid'), mel, bass, roots)
        print(f'midi/{name}.mid', len(mel), 'notes, bar 4:', ' '.join(librosa.midi_to_note(p, unicode=False) for _, _, p, _ in mel[-6:]))
