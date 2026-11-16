"""One command renders everything from a MIDI melody.

    music                         # = all: every sound (A/B set), every arrangement, the A/B player
    music sounds [name ...]       # A/B sounds only (names: preset file stems or parts of them)
    music arrange [name ...]      # arrangements only
    music check                   # verify each sound keeps the MIDI's timing and pitch
    music --midi other.mid ...    # drive everything from another MIDI file

Swap an instrument: edit presets/*.json (or copy one to make a new sound), or change a layer's "preset" in
arrangements/*.json, then rerun. Notes, timing and rhythm always come from the MIDI file.
"""
import argparse, glob, json, os, sys, time
import numpy as np, mido, soundfile as sf, pyloudnorm as pyln, pedalboard as pb
import engine
from engine import SR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'renders')
LAYER_LUFS = -20           # arrangement layer reference: gain keyframes are dB relative to this
CONTEXT_BASS, CONTEXT_BASS_DB = '16_808_bass', 4    # bass under every A/B sound, dB over the sound (a loud bass sets the key the melody is heard in)
SKIPPED = set()                                     # presets already reported as skipped
CENTS = {}                                          # tuning in use, for the player caption
PRE_LOOPS = 2              # loops rendered before the kept one, so tails wrap and the loop is seamless


# ---------------------------------------------------------------- inputs

def load_midi(path):
    mid = mido.MidiFile(path)
    tempo, num, tracks, cents = 500000, 4, {}, {}
    for tr in mid.tracks:
        name, tick, on, notes = None, 0, {}, []
        for msg in tr:
            tick += msg.time
            if msg.type == 'track_name': name = msg.name
            elif msg.type == 'set_tempo': tempo = msg.tempo
            elif msg.type == 'time_signature': num = msg.numerator
            elif msg.type == 'text' and msg.text.startswith('tuning_cents'):     # tuning_cents=  /  tuning_cents_bass=
                k, v = msg.text.split('='); cents['bass' if k.endswith('_bass') else 'melody'] = float(v)
            elif msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = (tick, msg.velocity)
            elif msg.type in ('note_off', 'note_on') and msg.note in on:
                s, v = on.pop(msg.note); notes.append((s, tick - s, msg.note, v))
        if notes: tracks[name or f'track{len(tracks)}'] = notes
    beat = tempo / 1e6 / mid.ticks_per_beat                      # seconds per tick
    bar_ticks = num * mid.ticks_per_beat
    end = max(s + d for ns in tracks.values() for s, d, _, _ in ns)
    loop_bars = int(np.ceil(end / bar_ticks - 1e-6))
    cents.setdefault('melody', 0.0); cents.setdefault('bass', cents['melody'])
    return {'cents': cents, 'bpm': 60e6 / tempo, 'bar_s': bar_ticks * beat, 'loop_bars': loop_bars, 'loop_s': loop_bars * bar_ticks * beat,
            'tracks': {k: sorted((s * beat, d * beat, p, v) for s, d, p, v in ns) for k, ns in tracks.items()}}


def load_dir(sub):
    items = []
    for f in sorted(glob.glob(os.path.join(HERE, sub, '*.json'))):
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        d['name'] = os.path.splitext(os.path.basename(f))[0]
        if sub == 'presets' and not engine.usable(d):      # optional Surge XT missing: skip, say so once
            if d['name'] not in SKIPPED: print(f'note: skipping {d["name"]} (Surge XT not installed, see README)'); SKIPPED.add(d['name'])
            continue
        items.append(d)
    return items


def pick(items, names):
    if not names: return items
    got = [i for i in items if any(n in i['name'] for n in names)]
    if not got and not any(n in k for n in names for k in SKIPPED): sys.exit(f'no match for {names}; have: {[i["name"] for i in items]}')
    return got


def cents_for(pre, song):
    """Tuning for a preset: the source's bass (bass, bass_root) and melody are tuned differently."""
    return song['cents']['bass' if pre.get('track', 'melody').startswith('bass') else 'melody']


def tile(notes, loop_s, t0, t1):
    """Repeat a loop's notes to cover [t0, t1) seconds, times relative to t0."""
    out = []
    for k in range(int(np.floor(t0 / loop_s)), int(np.ceil(t1 / loop_s)) + 1):
        out += [(s + k * loop_s - t0, d, p, v) for s, d, p, v in notes if t0 <= s + k * loop_s < t1]
    return out


# ---------------------------------------------------------------- loudness / output

def finish(x, lufs, ceiling_db=-1.0, measure=None):
    """Gain to target loudness (measured on `measure` slice), limit, hard ceiling. Returns (2, n)."""
    seg = x[:, measure] if measure is not None else x
    loud = pyln.Meter(SR).integrated_loudness(seg.T)
    if np.isfinite(loud):
        x = x * 10 ** ((lufs - loud) / 20)
    return engine.limit(x, ceiling_db)


def write(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x.T, SR, subtype='PCM_24')


# ---------------------------------------------------------------- sounds (A/B set)

def render_sound(pre, song):
    notes = song['tracks'][pre.get('track', 'melody')]
    L = song['loop_s']
    # seamless loop: render PRE_LOOPS+1 loops, keep the last one (it already holds the previous loop's tails).
    # Every pass is sample-identical (period=len(notes)); the first XF of the loop is crossfaded from the audio
    # just past its end, so the wrap is continuous even through modulated effects.
    XF = int(0.03 * SR)
    allnotes = tile(notes, L, 0, (PRE_LOOPS + 1) * L + 0.05)
    x = engine.render(allnotes, pre, (PRE_LOOPS + 1) * L + 0.05, period=len(notes), cents=cents_for(pre, song))
    i0, i1 = int(round(PRE_LOOPS * L * SR)), int(round((PRE_LOOPS + 1) * L * SR))
    x = finish(x, -16, measure=slice(i0, i1))
    loop = x[:, i0:i1].copy()
    ramp = np.linspace(0, 1, XF)
    loop[:, :XF] = loop[:, :XF] * ramp + x[:, i1:i1 + XF] * (1 - ramp)
    # full: two passes from silence with the natural ring-out
    tail = pre.get('tail', 6.0)
    full = finish(engine.render(tile(notes, L, 0, 2 * L), pre, 2 * L + tail, cents=cents_for(pre, song)), -16)
    return loop, full


def cmd_sounds(song, names):
    # A bare melody can read in a different key than the song it belongs to; the bass decides. So every
    # melody sound also gets a _with_bass.wav: the same loop over the context bass.
    ctx = {p['name']: p for p in load_dir('presets')}[CONTEXT_BASS]
    bass_loop = render_sound(ctx, song)[0] * 10 ** (CONTEXT_BASS_DB / 20)
    for pre in pick(load_dir('presets'), names):
        t = time.time()
        loop, full = render_sound(pre, song)
        write(os.path.join(OUT, 'sounds', f'{pre["name"]}_loop.wav'), loop)
        write(os.path.join(OUT, 'sounds', f'{pre["name"]}_full.wav'), full)
        if pre.get('track', 'melody') == 'melody':
            write(os.path.join(OUT, 'sounds', f'{pre["name"]}_with_bass.wav'), finish(loop + bass_loop, -16))
        print(f'  sound  {pre["name"]:<28} {time.time() - t:5.1f}s')
    # one file with every sound (over the bass) back to back, for A/B in any player or DAW
    loops = [os.path.join(OUT, 'sounds', f'{p["name"]}_with_bass.wav') for p in load_dir('presets') if p.get('track', 'melody') == 'melody']
    loops = [f for f in loops if os.path.exists(f)]
    if not loops: return
    gap = np.zeros((int(0.8 * SR), 2))
    seq = np.concatenate([np.concatenate([sf.read(f)[0], gap]) for f in loops])
    sf.write(os.path.join(OUT, 'ab_all_sounds.wav'), seq, SR, subtype='PCM_24')


# ---------------------------------------------------------------- arrangements

def keyframes(kf, bar_s, n, default):
    """[[bar, value], ...] -> per-sample array, linear between keys, held outside."""
    if not kf: return np.full(n, default, float)
    kf = sorted(kf)
    return np.interp(np.arange(n) / SR, [b * bar_s for b, _ in kf], [v for _, v in kf])


def layer_notes(lay, song, pre, a):
    """(layer start s, layer end s, notes relative to the layer start)."""
    bar = song['bar_s']
    t0, t1 = lay.get('from', 0) * bar, lay.get('to', a['bars']) * bar
    if 'notes' in lay:                                  # inline notes: [beat, beats, midi, vel] from the layer start
        q = bar / 4
        return t0, t1, [(b * q, d * q, m, v) for b, d, m, v in lay['notes']]
    return t0, t1, tile(song['tracks'][lay.get('track', pre.get('track', 'melody'))], song['loop_s'], t0, t1)


def write_midi(path, song, layers):
    """Arrangement as multitrack MIDI (one track per layer, named after its preset) for rebuilding in a DAW."""
    tpb = 480; spt = 60 / song['bpm'] / tpb
    mid = mido.MidiFile(ticks_per_beat=tpb)
    mid.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(song['bpm']), time=0),
                                      mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0),
                                      mido.MetaMessage('text', text=f'tuning_cents={song["cents"]["melody"]:.1f}', time=0),
                                      mido.MetaMessage('text', text=f'tuning_cents_bass={song["cents"]["bass"]:.1f}', time=0)]))
    for ch, (name, notes) in enumerate(layers):
        tr = mido.MidiTrack([mido.MetaMessage('track_name', name=name, time=0)]); mid.tracks.append(tr)
        ev = sorted([(round(s / spt), 1, p, v) for s, d, p, v in notes] + [(round((s + d) / spt), 0, p, 0) for s, d, p, v in notes])
        chan = ch + (ch >= 9)                     # skip channel 10 (GM drums)
        last = 0
        for tick, on, p, v in ev:
            tr.append(mido.Message('note_on' if on else 'note_off', note=p, velocity=v, channel=chan % 16, time=tick - last)); last = tick
    mid.save(path)


def render_layer(lay, song, presets, a):
    pre = presets[lay['preset']]
    t0, t1, notes = layer_notes(lay, song, pre, a)
    bar = song['bar_s']
    x = engine.render(notes, pre, t1 - t0 + lay.get('tail', a.get('tail', 8)), seed=lay.get('seed', 0), cents=cents_for(pre, song))
    loud = pyln.Meter(SR).integrated_loudness(x.T)
    if np.isfinite(loud):
        x *= 10 ** ((LAYER_LUFS - loud) / 20)          # gain 0 dB means the same loudness for every instrument
    n = x.shape[1]
    # automation is keyed in song bars; shift to layer time
    shift = lambda kf: [[b - lay.get('from', 0), v] for b, v in kf] if kf else kf
    x *= 10 ** (keyframes(shift(lay.get('gain')), bar, n, 0.0) / 20)
    if lay.get('lpf'):
        fc = keyframes(shift(lay['lpf']), bar, n, 20000)
        x = np.stack([engine.svf(c, fc, 0.8, 0, SR) for c in x])
    if lay.get('hpf'):
        fc = keyframes(shift(lay['hpf']), bar, n, 10)
        x = np.stack([engine.svf(c, fc, 0.7, 2, SR) for c in x])
    return int(round(t0 * SR)), x


def render_swell(sw, song, presets):
    """A reversed, reverb-drenched note of the melody that rises into a bar line: a transition built from the melody."""
    pre = presets[sw['preset']]
    q = song['bar_s'] / 4
    length = sw.get('beats', 4) * q
    x = engine.render([(0, q, sw.get('midi', song['tracks']['melody'][0][2]), 110)], pre, length + 2, cents=cents_for(pre, song))[:, ::-1]
    x = x[:, -int(length * SR):]
    x *= np.linspace(0, 1, x.shape[1]) ** 2 * 10 ** (sw.get('gain', -6) / 20)
    return int(round((sw['bar'] * song['bar_s'] - length) * SR)), x


def cmd_arrange(song, names):
    presets = {p['name']: p for p in load_dir('presets')}
    for a in pick(load_dir('arrangements'), names):
        sg = load_midi(os.path.join(HERE, a['midi'])) if 'midi' in a else song   # an arrangement may bring its own MIDI
        t = time.time()
        total = int((a['bars'] * sg['bar_s'] + a.get('tail', 8) + 2) * SR)
        mix = np.zeros((2, total)); stems = []
        parts = [(f'{i:02d}_{lay["preset"]}', *render_layer(lay, sg, presets, a)) for i, lay in enumerate(a['layers'])]
        parts += [(f'swell{i}_{sw["preset"]}', *render_swell(sw, sg, presets)) for i, sw in enumerate(a.get('swells', []))]
        for name, i0, x in parts:
            s = np.zeros_like(mix)
            k = min(x.shape[1], total - i0)
            s[:, i0:i0 + k] = x[:, :k] * 10 ** (presets[name.split('_', 1)[1]].get('mix_db', 0) / 20)
            mix += s; stems.append((name, s))
        bus = pb.Pedalboard([pb.HighpassFilter(25), pb.Compressor(threshold_db=-18, ratio=2, attack_ms=30, release_ms=300)])
        pre_gain = 0.3 / max(np.abs(mix).max(), 1e-9)
        out = finish(bus((mix * pre_gain).astype(np.float32), SR).astype(np.float64), a.get('lufs', -14))
        # trim trailing silence below -70 dBFS
        alive = np.where(np.abs(out).max(0) > 10 ** (-70 / 20))[0]
        out = out[:, :alive[-1] + int(0.2 * SR)] if len(alive) else out
        write(os.path.join(OUT, 'arrangements', f'{a["name"]}.wav'), out)
        lays = []
        for i, lay in enumerate(a['layers']):
            t0, _, ns = layer_notes(lay, sg, presets[lay['preset']], a)
            lays.append((f'{i:02d}_{lay["preset"]}', [(s + t0, d, p, v) for s, d, p, v in ns]))
        write_midi(os.path.join(OUT, 'arrangements', f'{a["name"]}.mid'), sg, lays)
        g = 10 ** ((a.get('lufs', -14) - pyln.Meter(SR).integrated_loudness((mix * pre_gain).T)) / 20)
        for name, s in stems:            # stems at mix level (pre master bus), same length: drop into any DAW at 0:00
            write(os.path.join(OUT, 'arrangements', f'{a["name"]}_stems', f'{name}.wav'), np.clip(s[:, :out.shape[1]] * pre_gain * g, -1, 1))
        print(f'  arrangement {a["name"]:<22} {out.shape[1] / SR:5.1f}s audio  {time.time() - t:5.1f}s')


# ---------------------------------------------------------------- A/B player

def cmd_player():
    sounds = [f'{p["name"]}_loop.wav' for p in load_dir('presets') if p.get('track', 'melody') == 'melody'
              and os.path.exists(os.path.join(OUT, 'sounds', f'{p["name"]}_loop.wav'))]
    arrs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(OUT, 'arrangements', '*.wav')))
    meta = {p['name']: p.get('description', '') for p in load_dir('presets')}
    meta.update({a['name']: a.get('description', '') for a in load_dir('arrangements')})
    rows = ''.join(f'<button class="s" data-i="{i}"><b>{i + 1 if i < 9 else ""}</b> {f[:-9]}<small>{meta.get(f[:-9], "")}</small></button>'
                   f'<audio loop preload="auto" src="sounds/{f}"></audio>' for i, f in enumerate(sounds))
    bass = f'<audio id="bass" loop preload="auto" src="sounds/{CONTEXT_BASS}_loop.wav"></audio>'
    arr = ''.join(f'<div class="a"><h3>{f[:-4]}</h3><small>{meta.get(f[:-4], "")}</small><audio controls preload="none" src="arrangements/{f}"></audio></div>' for f in arrs)
    src = os.path.join(HERE, 'source', 'original.wav')     # optional: a reference recording to compare against
    ref = (f'<h2>Reference</h2><audio controls preload="none" src="{os.path.relpath(src, OUT).replace(os.sep, "/")}"></audio>'
           if os.path.exists(src) else '')
    html = f"""<!doctype html><meta charset="utf-8"><title>Sound A/B</title>
<style>body{{background:#0b0b0d;color:#ddd;font:14px system-ui;max-width:900px;margin:24px auto;padding:0 16px}}
button.s{{display:block;width:100%;text-align:left;background:#16161a;color:#ddd;border:1px solid #2a2a30;padding:10px;margin:4px 0;border-radius:6px;cursor:pointer}}
button.s.on{{border-color:#c9a45c;background:#221d14}} small{{display:block;color:#888;margin-top:2px}} b{{color:#c9a45c;width:1.2em;display:inline-block}}
#go{{background:#c9a45c;color:#000;border:0;padding:10px 18px;border-radius:6px;font-weight:600;cursor:pointer}} .a{{margin:14px 0}} audio{{width:100%}}</style>
<h1>Sound A/B</h1><p>All loops play in sync from the same MIDI; click a sound (or press 1-9) to switch instantly. Loudness-matched at -16 LUFS.</p>
<p><label><input type="checkbox" id="bon" checked> 808 bass under every sound, as in the video (the key lives in the bass)</label></p>
<button id="go">Play / stop</button>{bass}{rows}<h2>Arrangements</h2>{arr}
{ref}
<script>const A=[...document.querySelectorAll('audio[loop]:not(#bass)')],S=[...A,bass],B=[...document.querySelectorAll('button.s')];let cur=0,on=false;
function sel(i){{cur=i;A.forEach((a,j)=>a.muted=j!==i);B.forEach((b,j)=>b.classList.toggle('on',j===i))}}
B.forEach((b,i)=>b.onclick=()=>{{sel(i);if(!on)go.onclick()}});sel(0);
A.forEach(a=>a.volume={10 ** (-CONTEXT_BASS_DB / 20):.3f});bass.muted=!bon.checked;bon.onchange=()=>bass.muted=!bon.checked;
go.onclick=()=>{{on=!on;if(on){{const t=A[cur].currentTime;S.forEach(a=>{{a.currentTime=t;a.play()}})}}else S.forEach(a=>a.pause())}};
document.onkeydown=e=>{{const i=+e.key-1;if(i>=0&&i<A.length)B[i].onclick()}};
setInterval(()=>{{if(on){{const t=A[cur].currentTime;S.forEach(a=>{{if(Math.abs(a.currentTime-t)>0.03)a.currentTime=t}})}}}},2000);</script>"""
    with open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------- check

def cmd_check(song, names):
    """Every sound must start its notes where the MIDI does and (for pitched engines) sound the MIDI pitch class."""
    import librosa
    notes = [n for n in song['tracks']['melody'] if n[0] < 2 * song['bar_s'] - 0.05]   # two bars is enough
    bad = 0
    for pre in pick(load_dir('presets'), names):
        if pre.get('track', 'melody') != 'melody': continue
        x = engine.render(notes, pre, 2 * song['bar_s'] + 1).mean(0)
        x = x / (np.abs(x).max() + 1e-9)
        on = librosa.onset.onset_detect(y=x, sr=SR, units='time', backtrack=True, delta=0.05)
        hit = np.mean([np.min(np.abs(on - s)) < 0.05 if len(on) else False for s, _, _, _ in notes])
        pc = []                                    # each pitch alone, so reverb from the previous note can't mislead
        for m in sorted({m for _, _, m, _ in notes}):
            y = engine.render([(0, 0.8, m, 100)], pre, 1.0, cents=cents_for(pre, song)).mean(0)[int(0.08 * SR):int(0.7 * SR)]
            f0, vf, _ = librosa.pyin(y, fmin=60, fmax=2500, sr=SR, frame_length=4096, hop_length=512)
            got = round(np.median(librosa.hz_to_midi(f0[vf]))) if vf.any() else None
            pc.append((m, got))
        pitch = np.mean([g is not None and g % 12 == m % 12 for m, g in pc])
        soft = pre.get('voice', {}).get('attack', 0) >= 0.05 or pre['engine'] == 'granular'
        ok = (hit >= 0.8 or soft) and (pitch == 1 or pre.get('inharmonic'))
        bad += not ok
        heard = ' '.join(librosa.midi_to_note(g, unicode=False) if g else '?' for _, g in pc)
        print(f'  {"ok  " if ok else "FAIL"} {pre["name"]:<24} onsets on grid {hit:4.0%}{" (soft attack)" if soft else "              "}'
              f'  pitches heard: {heard:<16}{" (inharmonic: informational)" if pre.get("inharmonic") else ""}')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', nargs='?', default='all', choices=['all', 'sounds', 'arrange', 'check', 'player'])
    ap.add_argument('names', nargs='*')
    ap.add_argument('--midi', default=os.path.join(HERE, 'midi', 'melody.mid'))
    ap.add_argument('--tune', type=float, help='tuning in cents from A440 (default: the tuning_cents in the MIDI, else 0)')
    a = ap.parse_args()
    global OUT
    stem = os.path.splitext(os.path.basename(a.midi))[0]
    if stem != 'melody': OUT = os.path.join(HERE, 'renders', stem)   # other MIDI files never overwrite the default's renders
    song = load_midi(a.midi)
    if a.tune is not None: song['cents'] = {'melody': a.tune, 'bass': a.tune}
    CENTS.update(song['cents'])
    engine.CTX['eighth'] = 30 / song['bpm']
    print(f'{os.path.basename(a.midi)}: tuning melody {CENTS["melody"]:+.1f} c / bass {CENTS["bass"]:+.1f} c, {song["bpm"]:.2f} BPM, {song["loop_bars"]}-bar loop ({song["loop_s"]:.3f}s), tracks {list(song["tracks"])}')
    if a.cmd == 'check': sys.exit(cmd_check(song, a.names))
    if a.cmd in ('all', 'sounds'): cmd_sounds(song, a.names)
    if a.cmd in ('all', 'arrange'): cmd_arrange(song, a.names)
    cmd_player()
    print('open', os.path.join(OUT, 'index.html'))


if __name__ == '__main__':
    main()
