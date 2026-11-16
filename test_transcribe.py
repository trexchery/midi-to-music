"""Round trip: midi/melody.mid -> audio (engine.render) -> transcribe.py (Basic Pitch, full mix) -> compare with the MIDI.

    python test_transcribe.py [preset-name ...]    # needs requirements-transcribe.txt; passes at >= 90% of notes (onset within 50 ms, same pitch)
"""
import json, os, subprocess, sys, tempfile
import soundfile as sf
import engine, music
from engine import SR

HERE = os.path.dirname(os.path.abspath(__file__))


def roundtrip(preset, tmp):
    song = music.load_midi(os.path.join(HERE, 'midi', 'melody.mid'))
    notes = song['tracks']['melody']
    pre = json.load(open(os.path.join(HERE, 'presets', preset + '.json'), encoding='utf-8')); pre['name'] = preset
    wav, mid = os.path.join(tmp, preset + '.wav'), os.path.join(tmp, preset + '.mid')
    sf.write(wav, engine.render(notes, pre, song['loop_s'] + 1).T, SR)
    subprocess.run([sys.executable, os.path.join(HERE, 'transcribe.py'), wav, '--no-split', '--out', mid], check=True, capture_output=True)
    got = music.load_midi(mid)['tracks']['melody']
    hit = sum(any(abs(s - gs) < 0.05 and p == gp for gs, _, gp, _ in got) for s, _, p, _ in notes)
    prec = sum(any(abs(s - gs) < 0.05 and p == gp for s, _, p, _ in notes) for gs, _, gp, _ in got)   # extras: reverb tails, harmonics
    return hit, len(notes), len(got), prec


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        ok = True
        for preset in sys.argv[1:] or ['01_glass_harmonica', '03_dark_tines', '07_analog_warmth']:
            hit, n, g, prec = roundtrip(preset, tmp)
            print(f'{preset}: {hit}/{n} notes matched ({hit / n:.0%}), transcribed {g} notes, {prec} of them true ({prec / g:.0%})')
            ok &= hit >= 0.9 * n
        sys.exit(not ok)
