"""Phase-based growl voices: 'fmstack' (DX/FM8-style operator stack), 'warp' (Serum/Vital-style phase warps on a
wavetable) and 'pd' (Casio CZ resonant phase distortion).

Pitch: every oscillator is a function of the note phase at f (integer op ratios, warps that map one cycle onto whole
table cycles), so the output is strictly harmonic of f. Only fmstack op ratios that are not integers (0.5, 1.5, ...)
add sub-harmonics; that is opt-in. Unison voices are detuned symmetrically around f (odd counts keep one voice on f).
Aliasing: fmstack and warp compute one cycle per 5.3 ms control block (the whole op stack / warp chain, at 2048
points), band-limit it by FFT to the playback pitch and play it back with crossfades (Vital's method: no aliasing by
construction, however high the index or hard the warp). A frame holds 1024 harmonics: FM or warps that spread past
that fold back onto other harmonics of f (pitch-safe timbre error: ~-40 dB on a 58 Hz warp default, down to ~-20 dB at
sync 8 + fm 8 or index 12 + feedback 1.5, vs a 16384-point frame). pd runs 4x oversampled. Everything is rendered at 4x and
decimated with a ~-100 dB filter; drive (tanh) runs at 8x.
Modulation LFOs are synced to engine.CTX['eighth'] and retrigger at note-on, which is on the eighth grid.
"""
import math, sys, time
from fractions import Fraction
import numpy as np, scipy.signal as ss
from numba import njit
import engine

SR = engine.SR; OS = 4; SRO = SR * OS
# decimator: ~-100 dB stopband from 22.6 kHz, so nothing the 4x stages make above it folds back into the audio band
_DEC = ss.firwin(383, 21000, window=('kaiser', 10.0), fs=SRO)


def _lfo(t, p, rng):
    """-1..1 LFO, period lfo_div eighths; 'sine' | 'saw' (falling) | 'square' | 'random' (stepped); 3 ms smoothing."""
    ph = t / (p.get('lfo_div', 2) * engine.CTX['eighth'])
    sh = p.get('lfo_shape', 'sine')
    if sh == 'sine': y = np.sin(2 * np.pi * ph)
    elif sh == 'saw': y = 1 - 2 * (ph % 1)
    elif sh == 'square': y = np.where(ph % 1 < 0.5, 1.0, -1.0)
    else: y = rng.uniform(-1, 1, int(ph[-1]) + 2)[ph.astype(int)]
    a = np.exp(-1 / (0.003 / (t[1] - t[0])))           # steps/edges in index or warp amount would click
    return ss.lfilter([1 - a], [1, -a], y, zi=[a * y[0]])[0]


def _down(x, p):
    """4x stereo -> optional tanh drive (oversampled, so its harmonics don't alias) -> SR -> optional lowpass."""
    d = p.get('drive', 0.0)
    if d > 0:                                            # wideband FM through tanh needs 8x, not 4x, to stay clean
        x = ss.resample_poly(np.tanh(d * ss.resample_poly(x, 2, 1, axis=1)) / np.tanh(d), 1, 2, axis=1)
    y = ss.resample_poly(x, 1, OS, axis=1, window=_DEC)
    if p.get('cutoff', 0):
        fc = np.full(y.shape[1], float(p['cutoff']))
        y = np.stack([engine.svf(c, fc, p.get('res', 0.9), 0, SR) for c in y])
    return y


def _bandlimit(g, fp, det):
    """Keep harmonics of the cycle below 20 kHz for the sharpest unison voice (fp = frame playback rate), drop DC."""
    X = np.fft.rfft(g, axis=1); X[:, 0] = 0; X[:, max(1, int(20000 / (fp * 2 ** (det / 1200)))) + 1:] = 0
    return np.fft.irfft(X, g.shape[1], axis=1)


def _unison_play(frames, fp, p, rng, n, u_def, u_max, det_def, sp_def):
    """Unison voices detuned symmetrically around fp (outer voices +-detune cents), equal-power spread, 4x stereo."""
    u = int(np.clip(p.get('unison', u_def), 1, u_max)); out = np.zeros((2, n * OS))
    for v in range(u):
        x = (v / (u - 1) * 2 - 1) if u > 1 else 0.0
        m = _play(frames, fp * 2 ** (x * p.get('detune', det_def) / 1200) / SRO, rng.random(), BLK * OS, n * OS)
        out += engine.pan(m, x * p.get('spread', sp_def)) / np.sqrt(u)
    return out


# ---------------------------------------------------------------- fmstack

@njit(cache=True)
def _fbcycle(beta, L):
    """Steady-state cycle of DX-style (two-sample averaged) self-feedback, settled over 16 cycles at L steps each."""
    y1 = 0.0; y2 = 0.0; out = np.empty(L)
    for c in range(16):
        for i in range(L):
            y = np.sin(2 * np.pi * i / L + beta * 0.5 * (y1 + y2)); y2 = y1; y1 = y; out[i] = y
    return out


def _opwave(fb):
    """None (plain sine) or the op's self-feedback cycle as a (1, 4096) table. Run per sample, feedback above 1 never
    settles (noise); one fixed cycle keeps it periodic, so harmonic."""
    return _fbcycle(fb, 4096)[None] if fb > 0 else None


FM_OPS = [{'ratio': 1, 'index': [1.5, 5.0, 0.35], 'feedback': 0.7},   # top: self-fed saw-ish growl source
          {'ratio': 2, 'index': [1.0, 3.0, 0.6]},                     # octave modulator: the vowel-ish bite
          {'ratio': 1, 'level': 1.0}]                                 # carrier


def v_fmstack(p, f, dur, vel, rng):
    """FM8/DX-style growl. Params:
    ops: list of <=4 ops, top (modulator) to bottom (carrier), each:
      ratio 1 (x f; integers keep it harmonic, 0.5/1.5 add a sub-octave), level 1 (output, carriers only),
      index [base, peak, decay_s] [0,0,1] (radians of PM it applies to its target; peak at note-on decays to base;
      updated every 5.3 ms, so decays under ~20 ms are softened),
      feedback 0 (self-feedback 0..1.5: sine -> saw-like at 1 -> hard buzz; shapes the op's own cycle, so it is
      exact DX feedback on top ops and a 'feedback waveform' on ops that are themselves modulated)
    algo 'serial' (ops chain into the last) | 'parallel' (two chains: first half, second half; carriers summed)
    lfo_div 2 (LFO period in eighths) · lfo_depth 0.3 (0..1, scales every index) · lfo_shape 'sine'|'saw'|'square'|'random'
    vel_mod 0.5 (0..1: low velocity lowers the index) · unison 3 (1-4) · detune 8 (cents, outer voices) · spread 0.6 (0..1)
    drive 0 (tanh, 0 = off, 1-4 = growl) · cutoff 0 (Hz lowpass, 0 = off) · res 0.9 (filter Q)
    attack 0.005 · decay 0.8 · sustain 0.7 · release 0.6 (amp ADSR, s / level)"""
    rel = p.get('release', 0.6); n_on = int(dur * SR); n = n_on + int(rel * SR)
    nb = n // BLK + 2; tb = np.arange(nb) * BLK / SR
    ops = p.get('ops', FM_OPS)[:4]; k = len(ops); h = (k + 1) // 2
    dst = [-1 if j == k - 1 or (p.get('algo', 'serial') == 'parallel' and j == h - 1) else j + 1 for j in range(k)]
    mod = (1 - p.get('vel_mod', 0.5) * (1 - vel)) * np.maximum(0, 1 + p.get('lfo_depth', 0.3) * _lfo(tb, p, rng))
    idx = [(b + (pk - b) * np.exp(-tb / max(dc, 1e-3))) * mod for b, pk, dc in (o.get('index', [0, 0, 1]) for o in ops)]
    ratio = [Fraction(o.get('ratio', 1)).limit_denominator(8) for o in ops]
    D = math.lcm(*[r.denominator for r in ratio])       # one frame spans D cycles of f (D = 1 for integer ratios)
    tabs = [_opwave(o.get('feedback', 0)) for o in ops]
    x = np.arange(LF) / LF; frames = np.empty((nb, LF))
    for c in range(0, nb, 256):
        s = slice(c, min(c + 256, nb)); m = [np.zeros((s.stop - c, 1))] * k; out = 0.0
        for j, o in enumerate(ops):
            ph = int(ratio[j] * D) * x + m[j] / (2 * np.pi)
            y = np.sin(2 * np.pi * ph) if tabs[j] is None else _read(tabs[j], np.zeros(s.stop - c), ph)
            if dst[j] >= 0: m[dst[j]] = m[dst[j]] + idx[j][s, None] * y
            else: out = out + o.get('level', 1.0) * y
        frames[s] = _bandlimit(out / dst.count(-1), f / D, p.get('detune', 8))
    return _down(_unison_play(frames, f / D, p, rng, n, 3, 4, 8, 0.6), p) *         engine.adsr(n_on, n, p.get('attack', 0.005), p.get('decay', 0.8), p.get('sustain', 0.7), rel)


# ---------------------------------------------------------------- warp

LF = 2048; BLK = 256                                    # frame length; control block (5.3 ms) between frames
_TABLES = {}


def _table(name):
    """(frames, L) single-cycle table; built-ins are exact Fourier series, peak 1."""
    if name not in _TABLES:
        if name.startswith('npy:'):
            _TABLES[name] = np.atleast_2d(np.load(name[4:])).astype(float)
        else:
            k = np.arange(1, 1024); X = np.zeros(1025, complex)
            X[1:1024] = -1j * {'saw': 1 / k, 'square': (k % 2) / k, 'tri': (k % 2) * (-1.0) ** ((k - 1) // 2) / k ** 2,
                               'sine': (k == 1) * 1.0}[name]
            w = np.fft.irfft(X, 2048); _TABLES[name] = (w / np.abs(w).max())[None]
    return _TABLES[name]


def _read(tab, pos, ph):
    """Table value at phase ph (rows: blocks) and frame position pos (frames, fractional), linear both ways."""
    F, L = tab.shape
    q = (ph % 1.0) * L; j = q.astype(int) % L; fq = q - np.floor(q); j1 = (j + 1) % L
    i0 = np.minimum(pos.astype(int), F - 1)[:, None]; i1 = np.minimum(i0 + 1, F - 1); fr = pos[:, None] - i0
    lin = lambda i: tab[i, j] + fq * (tab[i, j1] - tab[i, j])
    return lin(i0) if F == 1 else lin(i0) + fr * (lin(i1) - lin(i0))


def _warp(x, w, a):
    """One cycle x in [0,1) -> table phase. W(1)-W(0) is a whole number of cycles (sync aside), so it stays periodic."""
    m = w['mode']
    if m == 'sync': return x * np.clip(a, 1, 8)
    if m == 'bend': return x ** (4.0 ** -np.clip(a, -1, 1))
    if m == 'mirror':
        c = 1 - np.clip(a, 0, 1) / 2                     # turn point: 1 = no mirror, 0.5 = full palindrome
        return np.where(x < c, x / c, (1 - x) / np.maximum(1 - c, 1e-9))
    if m == 'remap':
        pts = sorted(w.get('points', [[0.5, 0.15]]))
        return x + np.clip(a, 0, 1) * (np.interp(x, [0] + [q[0] for q in pts] + [1], [0] + [q[1] for q in pts] + [1]) - x)
    if m == 'fm': return x + np.maximum(a, 0) / (2 * np.pi) * np.sin(2 * np.pi * int(w.get('ratio', 1)) * x)
    if m == 'pd':
        d = 0.5 * (1 - np.clip(a, 0, 0.98))
        return np.where(x < d, 0.5 * x / d, 0.5 + 0.5 * (x - d) / (1 - d))
    if m == 'quantize':
        s = np.round(a)
        return np.where(s >= 2, np.floor(x * np.maximum(s, 2)) / np.maximum(s, 2), x)
    raise ValueError(f'warp mode {m}')


@njit(cache=True)
def _play(frames, inc, ph, bl, N):
    """Read band-limited frames at phase ph, crossfading frame j -> j+1 across each bl-sample block."""
    out = np.empty(N); L = frames.shape[1]
    for i in range(N):
        j = i // bl; c = (i - j * bl) / bl
        q = ph * L; k0 = int(q) % L; fr = q - int(q); k1 = (k0 + 1) % L
        a = frames[j, k0] + fr * (frames[j, k1] - frames[j, k0])
        b = frames[j + 1, k0] + fr * (frames[j + 1, k1] - frames[j + 1, k0])
        out[i] = a + c * (b - a)
        ph += inc
        if ph >= 1.0: ph -= 1.0
    return out


WARP_DEF = [{'mode': 'sync', 'amount': 1.4, 'env': 2.6, 'decay': 0.4, 'lfo': 0.5},
            {'mode': 'bend', 'amount': 0.3}]


def v_warp(p, f, dur, vel, rng):
    """Serum/Vital-style warped wavetable. Params:
    table 'saw' ('saw'|'square'|'sine'|'tri' or 'npy:<path>' to a frames x 2048 array)
    pos 0 · pos_env 0 · pos_decay 0.5 · pos_lfo 0 (frame scan 0..1: base + note-on env decaying over pos_decay s + LFO)
    warps: 1-2 in series, each {mode, amount, env 0, decay 0.3 s, lfo 0 (LFO depth), +mode extras}; amount units:
      sync 1..8 (self-sync ratio; sync_window 1 = Hann-windowed formant sweep, 0 = hard buzz) · bend -1..1 ·
      mirror 0..1 · remap 0..1 (blend to 'points' [[x,y],...]) · fm index 0..~8 ('ratio' integer 1) ·
      pd 0..0.98 · quantize steps 2..64 (<2 = off)
    lfo_div 2 · lfo_shape 'sine' (eighth-synced, as fmstack) · vel_mod 0.5 (velocity scales the env depths)
    unison 5 (1-7) · detune 12 (cents) · spread 0.8 · drive 0 · cutoff 0 · res 0.9
    attack 0.01 · decay 0.8 · sustain 0.75 · release 0.6"""
    rel = p.get('release', 0.6); n_on = int(dur * SR); n = n_on + int(rel * SR)
    nb = n // BLK + 2; tb = np.arange(nb) * BLK / SR
    lfo = _lfo(tb, p, rng); vs = 1 - p.get('vel_mod', 0.5) * (1 - vel)
    env = lambda e, d: e * vs * np.exp(-tb / max(d, 1e-3))
    warps = p.get('warps', WARP_DEF)[:2]
    amts = [w.get('amount', 0) + env(w.get('env', 0), w.get('decay', 0.3)) + w.get('lfo', 0) * lfo for w in warps]
    tab = _table(p.get('table', 'saw'))
    pos = np.clip(p.get('pos', 0) + env(p.get('pos_env', 0), p.get('pos_decay', 0.5)) + p.get('pos_lfo', 0) * lfo, 0, 1) * (len(tab) - 1)
    x = np.arange(LF) / LF; frames = np.empty((nb, LF))
    for c in range(0, nb, 256):                         # chunks bound memory on long notes
        s = slice(c, min(c + 256, nb)); ph = np.broadcast_to(x, (s.stop - c, LF)); win = 1.0
        for w, a in zip(warps, amts):
            a = a[s, None]
            if w['mode'] == 'sync':
                win = win * (1 - p.get('sync_window', 1.0) * np.clip(a - 1, 0, 1) * np.cos(np.pi * x) ** 2)
            ph = _warp(ph % 1.0, w, a)
        frames[s] = _bandlimit(_read(tab, pos[s], ph) * win, f, p.get('detune', 12))
    return _down(_unison_play(frames, f, p, rng, n, 5, 7, 12, 0.8), p) * engine.adsr(n_on, n, p.get('attack', 0.01), p.get('decay', 0.8), p.get('sustain', 0.75), rel)


# ---------------------------------------------------------------- pd

_PDWIN = {'saw': lambda x: 1 - x, 'tri': lambda x: 1 - np.abs(2 * x - 1), 'trap': lambda x: np.minimum(1, 2 * (1 - x))}


def v_pd(p, f, dur, vel, rng):
    """Casio CZ resonant phase distortion: window(ph) * (1 - cos(2pi r ph))/2, r = resonance as a multiple of f. Params:
    window 'saw' ('saw' bright reso-saw | 'tri' hollow | 'trap' thick reso-pulse)
    r_start 8 · r_end 2 (resonance ratio at note-on -> settles to; non-integer is fine, the output stays harmonic;
      capped at 20 kHz / f)
    glide 0.25 (s, time constant of the r sweep) · retrig 0 (eighths: restart the sweep every N eighths, 0 = once)
    vel_mod 0.5 (low velocity shortens the sweep range) · sub_level 0.3 (clean sine at f, added after drive)
    drive 0 · cutoff 0 · res 0.9 · attack 0.003 · decay 0.6 · sustain 0.75 · release 0.5"""
    rel = p.get('release', 0.5); n_on = int(dur * SR); n = n_on + int(rel * SR); N = n * OS
    t = np.arange(N) / SRO
    re_ = p.get('r_end', 2.0); rs = re_ + (p.get('r_start', 8.0) - re_) * (1 - p.get('vel_mod', 0.5) * (1 - vel))
    rt = p.get('retrig', 0) * engine.CTX['eighth']
    r = re_ + (rs - re_) * np.exp(-(t % rt if rt > 0 else t) / max(p.get('glide', 0.25), 1e-3))
    if rt > 0:
        a = np.exp(-1 / (0.002 * SRO))                   # soften the jump back to r_start: it would click
        r = ss.lfilter([1 - a], [1, -a], r, zi=[a * r[0]])[0]
    r = np.minimum(r, 20000 / f)                         # a resonance above hearing only aliases at 4x (high notes)
    ph0 = rng.random(); ph = (np.arange(N) * (f / SRO) + ph0) % 1.0
    wf = _PDWIN[p.get('window', 'saw')]
    y = wf(ph) * (0.5 - 0.5 * np.cos(2 * np.pi * r * ph))
    # subtract each cycle's exact mean (a function of r): the amp envelope would otherwise turn the DC into a thump
    rg = np.linspace(r.min(), r.max() + 1e-9, 64); xg = (np.arange(2048) + 0.5) / 2048
    y -= np.interp(r, rg, (wf(xg) * (0.5 - 0.5 * np.cos(2 * np.pi * rg[:, None] * xg))).mean(1))
    out = _down(np.stack([y, y]), p)
    out += p.get('sub_level', 0.3) * np.sin(2 * np.pi * (f * np.arange(n) / SR + ph0))
    return out * engine.adsr(n_on, n, p.get('attack', 0.003), p.get('decay', 0.6), p.get('sustain', 0.75), rel)


VOICES = {'fmstack': v_fmstack, 'warp': v_warp, 'pd': v_pd}


# ---------------------------------------------------------------- selftest

def _pitch(x, f):
    """Peak within +-50 c of f: Blackman-Harris, 8x zero padding, parabolic interpolation on log magnitude."""
    seg = x.sum(0)[int(0.4 * SR):int(4.4 * SR)]           # 4 s: LFO sidebands (2.8 Hz) resolved even at 58 Hz
    nf = 8 * 2 ** int(np.ceil(np.log2(len(seg)))); S = np.abs(np.fft.rfft(seg * ss.windows.blackmanharris(len(seg)), nf))
    df = SR / nf; lo = int(f * 2 ** (-50 / 1200) / df); k = lo + np.argmax(S[lo:int(f * 2 ** (50 / 1200) / df) + 1])
    a, b, c = np.log(S[k - 1:k + 2] + 1e-300)
    return 1200 * np.log2((k + 0.5 * (a - c) / (a - 2 * b + c)) * df / f)


def _inharm(x, f):
    """dB of energy farther than 2% of f from every harmonic of f, on 2 s of sustain."""
    seg = x.sum(0)[int(1.0 * SR):int(3.0 * SR)]
    P = np.abs(np.fft.rfft(seg * ss.windows.blackmanharris(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / SR); d = np.abs(fr / f - np.round(fr / f)) * f
    return 10 * np.log10(P[d > 0.02 * f].sum() / P.sum() + 1e-30)


def selftest():
    ok = True

    def check(name, cond, info=''):
        nonlocal ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {info}")

    # fmstack cfg1 runs without LFO: its index sweeps through a Bessel zero of the fundamental, so the f partial is
    # amplitude-modulated through zero and the LFO sidebands (harmonic +-2.8 Hz) outweigh it; that is timbre, not pitch
    pitch_cfgs = {'fmstack': [{}, {'algo': 'parallel', 'lfo_depth': 0, 'ops': [{'ratio': 3, 'index': [2, 4, .3], 'feedback': 1.0},
                                                               {'ratio': 1}, {'ratio': 1, 'index': [1, 2, .5]}, {'ratio': 2, 'level': .5}]}],
                  'warp': [{}, {'table': 'square', 'warps': [{'mode': 'fm', 'amount': 3, 'ratio': 2}, {'mode': 'quantize', 'amount': 6}]},
                           {'table': 'tri', 'warps': [{'mode': 'mirror', 'amount': 0.7, 'lfo': 0.3}, {'mode': 'pd', 'amount': 0.6}]},
                           {'warps': [{'mode': 'remap', 'amount': 1, 'points': [[0.3, 0.7]]}]}],
                  'pd': [{}, {'window': 'tri', 'retrig': 1, 'r_start': 11.3}, {'window': 'trap', 'r_end': 3.7}]}
    aggressive = {'fmstack': {'ops': [{'ratio': 1, 'index': [5, 9, .3], 'feedback': 1.2}, {'ratio': 2, 'index': [3, 6, .5]},
                                      {'ratio': 1}], 'lfo_div': 16, 'lfo_depth': 0.5, 'drive': 3, 'unison': 1},
                  'warp': {'warps': [{'mode': 'sync', 'amount': 7, 'sync_window': 0}, {'mode': 'quantize', 'amount': 5}],
                           'lfo_div': 16, 'drive': 3, 'unison': 1},
                  'pd': {'r_start': 16, 'r_end': 14, 'drive': 3, 'sub_level': 0}}
    for name, fn in VOICES.items():
        y = fn({}, 308.87, 1.0, 0.8, np.random.default_rng(1))
        check(f'{name} shape/finite', y.shape == (2, SR + int(0.6 * SR) if name != 'pd' else SR + int(0.5 * SR))
              and np.isfinite(y).all(), f'peak {np.abs(y).max():.2f}')
        for ci, cfg in enumerate(pitch_cfgs[name]):
            for f in (308.87, 462.78, 58.11):
                c = fn({**cfg, 'unison': 1}, f, 4.6, 0.8, np.random.default_rng(2))
                e = _pitch(c, f); check(f'{name} cfg{ci} pitch {f} Hz', abs(e) < 1, f'{e:+.3f} c')
        db = _inharm(fn(aggressive[name], 462.78, 3.2, 1.0, np.random.default_rng(3)), 462.78)
        check(f'{name} aliasing (aggressive, 462.78 Hz)', db < -60, f'{db:.1f} dB')
        if name == 'pd':                                 # high note, resonance far above hearing
            db = _inharm(fn({'r_start': 40, 'r_end': 24, 'drive': 3, 'sub_level': 0}, 1850.0, 3.2, 1.0, np.random.default_rng(3)), 1850.0)
            check('pd aliasing (r 40->24, 1850 Hz)', db < -60, f'{db:.1f} dB')
        q = {'lfo_shape': 'random'}
        a = fn(q, 100.0, 0.5, 0.7, np.random.default_rng(5)); b = fn(q, 100.0, 0.5, 0.7, np.random.default_rng(5))
        check(f'{name} determinism', np.array_equal(a, b))
        t0 = time.perf_counter(); fn({}, 308.87, 1.0, 0.8, np.random.default_rng(4)); dt = time.perf_counter() - t0
        check(f'{name} speed (1 s note)', dt < 0.5, f'{dt * 1000:.0f} ms')
    y = engine.render([(0, 0.36, 70, 100), (0.36, 0.36, 63, 90)], {'name': 't', 'engine': 'fmstack'}, 1.5)
    check('engine registration (render fmstack)', y.shape == (2, int(1.5 * SR)) and np.isfinite(y).all())
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    selftest()
