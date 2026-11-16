"""Dynamics and drive effects: OTT 3-band up/down compression, wavefolder, multiband distortion, clip stack,
note-retriggered lowpass gate and grid-locked filter/gain sweep. All are memoryless across calls (output is a function
of the input buffer and engine.CTX only), so loops render sample-identical. None of them moves pitch: every nonlinear
stage runs 4-32x oversampled (raised automatically with drive) through a steep 22 kHz anti-alias FIR; crossovers and
swept filters are zero-phase, so a hard onset gets a short pre-echo (ott/mbdist: under -40 dB beyond 3-7 ms before it).
"""
import sys, time
from functools import lru_cache
import numpy as np, scipy.signal as ss
from numba import njit
from scipy.ndimage import maximum_filter1d
import engine

SR = engine.SR


# ---------------------------------------------------------------- helpers

def _coef(ms, sr=SR): return 1.0 if ms <= 0 else 1 - np.exp(-1 / (ms / 1000 * sr))


@njit(cache=True)
def _follow(x, att, rel):
    """One-pole envelope follower: coefficient att while rising, rel while falling."""
    y = np.empty_like(x); e = 0.0
    for i in range(x.shape[0]):
        e += (att if x[i] > e else rel) * (x[i] - e)
        y[i] = e
    return y


@njit(cache=True)
def _comp(x, thr, ratio, att, rel, mk):
    """Stereo-linked feed-forward peak compressor on a (2, n) array, then tanh(mk * .) (fused: no temporaries at 32x)."""
    out = np.empty_like(x); e = 0.0; s = 1 - 1 / ratio
    for i in range(x.shape[1]):
        a = max(abs(x[0, i]), abs(x[1, i]))
        e += (att if a > e else rel) * (a - e)
        g = mk if e <= thr else mk * (thr / e) ** s
        out[0, i] = 1 - 2 / (np.exp(2 * x[0, i] * g) + 1)     # = tanh, 2x faster in numba (err 3e-16)
        out[1, i] = 1 - 2 / (np.exp(2 * x[1, i] * g) + 1)
    return out


def _lp0(x, fc):
    """Zero-phase Butterworth-2 forward+backward = LR4 magnitude; x - _lp0(x) is the matching LR4 highpass."""
    return ss.sosfiltfilt(ss.butter(2, fc, fs=SR, output='sos'), x, axis=-1, padlen=min(9, x.shape[-1] - 1))


def _split3(x, f1, f2):
    lo = _lp0(x, f1); r = x - lo; mid = _lp0(r, f2)
    return lo, mid, r - mid                     # lo + mid + high == x exactly


@lru_cache(None)
def _aa(k):
    """Anti-alias FIR at k*SR: flat to 20 kHz, -100 dB from 24 kHz (scipy's default kaiser(5) stops only ~-55 dB)."""
    n, beta = ss.kaiserord(100, 4000 / (SR * k / 2))
    return ss.firwin(n | 1, 22000, window=('kaiser', beta), fs=SR * k)


@lru_cache(None)
def _aa2(r):
    """Loose FIR for the 4x -> 4r*x stage: passes 24 kHz, stops -100 dB from 168 kHz (images/aliases of 0-24 kHz)."""
    fs = 4 * SR * r; n, beta = ss.kaiserord(100, 144000 / (fs / 2))
    return ss.firwin(n | 1, 96000, window=('kaiser', beta), fs=fs)


def _os(fn, x, k=4):
    """Run fn on x oversampled k times (4 | k); fn gets (2, k*n) and must keep the shape. The steep 22 kHz FIR runs only
    at 4x (its length grows with the rate); the rest is a short FIR, ~4x cheaper at 32x."""
    r = k // 4
    u = ss.resample_poly(x, 4, 1, axis=-1, window=_aa(4))
    u = fn(ss.resample_poly(u, r, 1, axis=-1, window=_aa2(r)) if r > 1 else u)
    return ss.resample_poly(ss.resample_poly(u, 1, r, axis=-1, window=_aa2(r)) if r > 1 else u, 1, 4, axis=-1, window=_aa(4))


def _match(y, ref):
    """Scale y to ref's RMS so drive/depth change tone, not loudness."""
    return y * np.sqrt((ref ** 2).mean() / max((y ** 2).mean(), 1e-30))


def _since(n):
    """Seconds since the latest note start in engine.CTX['notes'] per sample (inf before the first note)."""
    st = np.sort(np.round(np.array([s for s, *_ in engine.CTX['notes']] or [np.inf]) * SR))   # sample grid, as render()
    i = np.searchsorted(st, np.arange(n), 'right') - 1
    return np.where(i >= 0, (np.arange(n) - st[np.maximum(i, 0)]) / SR, np.inf)


def _svf0(x, fc, q, mode, fb=None):
    """Zero-phase engine.svf (forward, then backward with cutoff fb): a fast sweep would otherwise rotate each
    harmonic's phase and bend its pitch for a moment; forward+backward cancels the phase, slopes double.
    Each pass gets sqrt(q), so q stays the resonant peak gain as on a single engine.svf; bp is normalized to unity."""
    q = np.sqrt(q)
    s = lambda c, f: engine.svf(np.ascontiguousarray(c), np.ascontiguousarray(f), q, mode, SR) / (q if mode == 1 else 1)
    fb = fc if fb is None else fb
    return np.stack([s(s(c, fc)[::-1], fb[::-1])[::-1] for c in x])


def _lfo(n, div, shape, phase=0.0):
    """0..1 LFO locked to the MIDI grid: one cycle per `div` eighths, phase 0 at time 0."""
    ph = (np.arange(n) / SR / (div * engine.CTX['eighth']) + phase) % 1.0
    return {'sine': 0.5 - 0.5 * np.cos(2 * np.pi * ph), 'tri': 1 - np.abs(2 * ph - 1), 'saw': ph, 'ramp': 1 - ph,
            'square': 0.5 + 0.5 * np.tanh(6 * np.sin(2 * np.pi * ph)) / np.tanh(6)}[shape]


CURVES = {'tanh': np.tanh, 'hard': lambda u: np.clip(u, -1, 1), 'fold': lambda u: np.sin(0.5 * np.pi * u),
          'asym': lambda u: np.tanh(u + 0.4) - np.tanh(0.4)}     # asym: offset knee -> even harmonics (DC removed after)


# ---------------------------------------------------------------- effects

def fx_ott(x, p, rng):
    """OTT-style 3-band upward + downward compression: tails and grit pulled up, peaks squashed, wet level-matched.
    xover_lo 120 Hz (40-400), xover_hi 2500 Hz (1k-8k); per-band params take a scalar or [low, mid, high];
    depth 0.4 (0-1 dry/wet); in_db 0 (-24..24)."""
    bp = lambda k, d: np.broadcast_to(np.asarray(p.get(k, d), float), (3,))   # scalar or [low, mid, high]
    xi = x * 10 ** (p.get('in_db', 0) / 20)                       # in_db: drive into the detectors (harder squash)
    att, rel = bp('attack_ms', (8, 4, 2)), bp('release_ms', (160, 90, 50))   # slower lows: no ripple on sub cycles
    dt, dr = bp('down_thr_db', (-33.8, -30.2, -35.5)), bp('down_ratio', 66)  # Xfer OTT's stock thresholds/ratios
    ut, ur = bp('up_thr_db', (-40.8, -41.8, -40.8)), bp('up_ratio', 4.17)
    umax, gdb = bp('up_max_db', 18), bp('gain_db', 0)             # upward gain clamp; band balance (wet level is matched)
    wet = np.zeros_like(x)

    def band(u, j):                                               # at 4x: fast gain moves would alias at 1x
        lv = 20 * np.log10(_follow(np.abs(u).max(0), _coef(att[j], 4 * SR), _coef(rel[j], 4 * SR)) + 1e-12)
        g = np.minimum(dt[j] - lv, 0) * (1 - 1 / dr[j]) + np.minimum(np.maximum(ut[j] - lv, 0) * (1 - 1 / ur[j]), umax[j])
        la = min(int(3 * att[j] / 1000 * 4 * SR), len(g))         # look-ahead 3 attack constants: no boosted onset spike
        g = np.concatenate([g[la:], np.full(la, g[-1])])
        return u * (10 ** ((g + gdb[j]) / 20) - 1)               # only the gain change is resampled: ratio 1 stays exact
    for j, b in enumerate(_split3(xi, p.get('xover_lo', 120), p.get('xover_hi', 2500))):
        wet += b + _os(lambda u: band(u, j), b, 4)
    d = p.get('depth', 0.4)
    return (1 - d) * x + d * _match(wet, x)


def fx_fold(x, p, rng):
    """Sine wavefolder (Buchla-ish when stages > 1), 8x oversampled, with grid-synced and note-retriggered drive.
    drive_db 12 (0-36), bias 0 (-1..1), stages 1 (1-4), lfo_db 0 (0-24) div 2 eighths shape sine|tri|saw|ramp|square,
    env_db 0 (0-24) env_decay 0.15 s (0.02-1), mix 1 (0-1), os 8 (4-32, minimum: raised with drive and stages).
    Aliasing stays under -60 dB (saw to 19 kHz, up to 1850 Hz) for peak drive (drive+lfo+env) up to 36 dB at 1-2 stages,
    30 dB at 3 and 24 dB at 4; beyond that the 32x cap lets grit through (stages 3-4 are the noisy, inharmonic end)."""
    n = x.shape[1]
    st = max(1, int(p.get('stages', 1)))                          # stages: drive split over cascaded folds
    env = _follow(np.exp(-_since(n) / p.get('env_decay', 0.15)), _coef(1.0), 1.0)   # 1 ms rise: a drive jump clicks
    ddb = p.get('drive_db', 12) + p.get('lfo_db', 0) * _lfo(n, p.get('div', 2), p.get('shape', 'sine')) + p.get('env_db', 0) * env
    gs = 10 ** (ddb / 20 / st)
    bw = (gs.max() * np.pi / 2 + 1) ** st                         # rough fold bandwidth growth (Carson per stage)
    k = int(min(32, max(p.get('os', 8), 2 ** np.ceil(np.log2(bw / 8)))))
    g = np.repeat(gs, k)
    b = p.get('bias', 0.0)                                        # bias: asymmetry -> even harmonics

    def fold(u):
        u = np.sin(0.5 * np.pi * (g * u + b))
        for _ in range(st - 1):
            u = np.sin(0.5 * np.pi * g * u)
        return u
    # minus the zero-input output: silence stays silent (no bias DC step for the HPF to thump on); HPF: signal DC
    y = ss.sosfilt(ss.butter(2, 20, 'highpass', fs=SR, output='sos'), _os(lambda u: fold(u) - fold(0 * u), x, k))
    m = p.get('mix', 1.0)
    return (1 - m) * x + m * _match(y, x)


def fx_mbdist(x, p, rng):
    """Multiband distortion: clean mono sub under xover_lo, independently driven mid/high bands (oversampled).
    xover_lo 150 Hz (60-400), xover_hi 2500 Hz (1k-8k), low_db 0; mid_drive_db 15 / high_drive_db 9 (0-36);
    mid_curve asym / high_curve tanh (tanh|hard|fold|asym); mid_db 0 / high_db -3 (-24..12); mix 1;
    os 4 (4|8|16|32, minimum: raised per band to 8x above 18 dB drive, 16x above 24 dB; hard: 8x, 16x above 18 dB),
    so aliasing stays < -60 dB."""
    lo, mid, hi = _split3(x, p.get('xover_lo', 150), p.get('xover_hi', 2500))
    wet = np.broadcast_to(lo.mean(0), x.shape) * 10 ** (p.get('low_db', 0) / 20)   # sub safety: mono, untouched
    for b, nm, d_db, crv, l_db in ((mid, 'mid', 15, 'asym', 0), (hi, 'high', 9, 'tanh', -3)):
        d, cn = p.get(f'{nm}_drive_db', d_db), p.get(f'{nm}_curve', crv)
        g, c = 10 ** (d / 20), CURVES[cn]
        k = max(p.get('os', 4), 16 if d > 24 or (cn == 'hard' and d > 18) else 8 if d > 18 or cn == 'hard' else 4)
        y = _os(lambda u: c(g * u), b, k)
        y = y - _lp0(y, p.get('xover_lo', 150))                   # no intermod/DC leaking under the sub band
        wet = wet + _match(y, b) * 10 ** (p.get(f'{nm}_db', l_db) / 20)
    m = p.get('mix', 1.0)
    return (1 - m) * x + m * wet


def fx_clipstack(x, p, rng):
    """'Sine compression': n passes of fast compressor -> makeup -> soft clip, all at 8x, level-matched out.
    passes 3 (1-6), drive_db 9 (0-30, total over passes), thr_db -12 (-40..0), ratio 4 (1-20), attack_ms 1 (0.1-10),
    release_ms 40 (5-300), mix 1 (0-1), os 8 (4-32, minimum: raised to 16x/32x as drive_db + 2*passes passes 24/34,
    so aliasing stays < -60 dB; the heaviest settings cost about 1 s per second of audio)."""
    npass = max(1, int(p.get('passes', 3)))
    s = p.get('drive_db', 9) + 2 * npass                          # near-hard clipping: alias grows with drive and passes
    k = max(p.get('os', 8), 8 if s <= 24 else 16 if s <= 34 else 32)
    thr, ratio = 10 ** (p.get('thr_db', -12) / 20), p.get('ratio', 4.0)
    mk = thr ** -(1 - 1 / ratio) * 10 ** (p.get('drive_db', 9) / 20 / npass)   # undo the squash, then push the clip
    att, rel = _coef(p.get('attack_ms', 1.0), SR * k), _coef(p.get('release_ms', 40), SR * k)

    def stack(u):
        for _ in range(npass):
            u = _comp(u, thr, ratio, att, rel, mk)
        return u
    m = p.get('mix', 1.0)
    return (1 - m) * x + m * _match(_os(stack, x, k), x)


def fx_lpg(x, p, rng):
    """Lowpass gate: every note start snaps cutoff and level open, both decay: a pluck/bloom on each eighth.
    decay_s 0.28 (0.02-2), min_hz 120 (20-2000), max_hz 7000 (500-20000), res 1.2 (0.5-8, peak gain at cutoff),
    vca 0.7 (0-1, how far the level closes), attack_ms 1 (0-20).
    Inharmonic option: res above ~4 with decay_s under ~0.05 rings as a falling filter 'ping' (a chirp not at f)."""
    n = x.shape[1]
    e = np.exp(-_since(n) / p.get('decay_s', 0.28))
    e = _follow(e, _coef(p.get('attack_ms', 1.0)), 1.0)          # ~1 ms rise: instant to the ear, no click
    lo, hi = p.get('min_hz', 120), p.get('max_hz', 7000)
    fc = lo * (hi / lo) ** e
    fb = maximum_filter1d(fc, 97, origin=-48)                     # backward pass opens 2 ms early: no pre-echo
    return _svf0(x, fc, p.get('res', 1.2), 0, fb) * (1 - p.get('vca', 0.7) * (1 - e))


def fx_msweep(x, p, rng):
    """Grid-locked movement: lp/bp/hp sweep or gain pump, one cycle per `div` eighths (the wub rate).
    mode lp|bp|hp|gain, div 2 (0.25-16), shape sine|tri|saw|ramp|square, depth 0.8 (0-1), phase 0 (0-1 cycle),
    min_hz 150 / max_hz 6000 (20-20000), res 1.5 (0.5-8)."""
    n, mode = x.shape[1], p.get('mode', 'lp')
    u = 1 - p.get('depth', 0.8) * (1 - _lfo(n, p.get('div', 2), p.get('shape', 'sine'), p.get('phase', 0.0)))
    u = _follow(u, _coef(1.5), _coef(1.5))                        # de-click saw/ramp/square edges
    if mode == 'gain':
        return x * u
    lo, hi = p.get('min_hz', 150), p.get('max_hz', 6000)
    fc = lo * (hi / lo) ** (1 - u if mode == 'hp' else u)        # u = openness; hp opens downward
    return _svf0(x, fc, p.get('res', 1.5), {'lp': 0, 'bp': 1, 'hp': 2}[mode])


FX = {'ott': fx_ott, 'fold': fx_fold, 'mbdist': fx_mbdist, 'clipstack': fx_clipstack, 'lpg': fx_lpg,
      'msweep': fx_msweep}


# ---------------------------------------------------------------- selftest

def _saw(f, sec, top=12000):
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, int(top / f) + 1))
    y *= 0.5 / np.abs(y).max()
    return np.stack([y, y])


def _peak(y, f):
    """Peak frequency near f: Hann, 8x zero padding, parabolic interpolation on log magnitude."""
    nf = 8 * len(y); s = np.log(np.abs(np.fft.rfft(y * np.hanning(len(y)), nf)) + 1e-30); df = SR / nf
    lo, hi = int(f * 0.97 / df), int(f * 1.03 / df) + 1
    i = lo + int(np.argmax(s[lo:hi])); a, b, c = s[i - 1:i + 2]
    return (i + 0.5 * (a - c) / (a - 2 * b + c)) * df


def _period_cents(y, f):
    """Pitch error from periodicity: the shift (scan +-5 c, parabolic) that best maps y onto itself one cycle later.
    Robust to rhythmic modulation, which only adds sidebands around the harmonics an FFT peak-pick can trip on."""
    Y = np.fft.rfft(y); w = 2j * np.pi * np.fft.rfftfreq(len(y), 1 / SR); s = slice(SR // 10, -SR // 10)
    c = np.linspace(-5, 5, 41)
    r = [((y - np.fft.irfft(Y * np.exp(-w / f * 2 ** (ci / 1200)), len(y)))[s] ** 2).sum() for ci in c]
    i = int(np.clip(np.argmin(r), 1, 39)); a, b, d = r[i - 1:i + 2]
    return c[i] + 0.25 * (a - d) / (a - 2 * b + d)


def _alias_db(y, f):
    P = np.abs(np.fft.rfft(y * ss.windows.blackmanharris(len(y)))) ** 2
    fr = np.fft.rfftfreq(len(y), 1 / SR); h = fr / f
    return 10 * np.log10(P[np.abs(h - np.round(h)) * f > 0.02 * f].sum() / P.sum())


def selftest():
    fails = 0

    def check(name, ok, info=''):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")

    def notes(f, sec):                                            # an eighth-note line at f, like the real MIDI
        m = 69 + 12 * np.log2(f / 440); e = engine.CTX['eighth']
        engine.CTX['notes'] = [(i * e, e * 0.9, m, 100) for i in range(int(sec / e) + 1)]

    saved = dict(engine.CTX)
    var = {'ott': [dict(depth=1, in_db=12)], 'fold': [dict(drive_db=24, stages=2, bias=0.3), dict(lfo_db=12, env_db=12)],
           'mbdist': [dict(mid_drive_db=24, mid_curve='hard', high_drive_db=18, high_curve='fold')],
           'clipstack': [dict(drive_db=24, passes=4)], 'lpg': [dict(decay_s=0.08, res=4, max_hz=16000)],
           'msweep': [dict(mode='bp', shape='saw', div=1, depth=1, res=4), dict(mode='gain', shape='square', div=0.5, depth=1)]}
    for nm, fn in FX.items():                                     # var[nm][0] = aggressive settings for the alias check
        for vi, p in enumerate([{}] + var[nm]):
            tag = nm + (f'/v{vi} {p}' if p else ' default')
            for f in (308.87, 462.78, 58.11):
                notes(f, 4.0); x = _saw(f, 4.0)
                y = fn(x, p, np.random.default_rng(0))
                if f == 308.87:
                    check(f'{tag} shape/finite', y.shape == x.shape and np.isfinite(y).all(), f'peak {np.abs(y).max():.2f}')
                seg = y.mean(0)[SR // 2:]
                errs = [1200 * np.log2(_peak(seg, h * f) / (h * f)) for h in (1, 2, 3)]
                info = 'cents h1-3 ' + ' '.join(f'{c:+.3f}' for c in errs)
                if nm == 'fold' and vi == 2:                      # drive sweep flips Bessel weights: sidebands > carrier
                    print(f'INFO  {tag} FFT peaks {f} Hz (sidebands, not pitch)  {info}')
                else:
                    check(f'{tag} pitch {f} Hz FFT', max(map(abs, errs)) < 1, info)
                c = _period_cents(seg, f)
                check(f'{tag} pitch {f} Hz period', abs(c) < 1, f'{c:+.3f} cents')
        if nm in ('lpg', 'msweep'):
            print(f'SKIP  {nm} aliasing: linear time-varying filter/gain, no nonlinearity to alias')
        else:
            notes(462.78, 1.5); x = _saw(462.78, 1.5)
            a = _alias_db(fn(x, var[nm][0], np.random.default_rng(0)).mean(0)[SR // 2:SR // 2 + SR], 462.78)
            check(f'{nm} aliasing @462.78 {var[nm][0]}', a < -60, f'{a:.1f} dB (input {_alias_db(x[0, SR // 2:SR // 2 + SR], 462.78):.1f})')
        notes(308.87, 2.0); x = _saw(308.87, 2.0)
        check(f'{nm} deterministic', np.array_equal(fn(x, var[nm][0], np.random.default_rng(5)),
                                                    fn(x, var[nm][0], np.random.default_rng(5))))
        x = _saw(308.87, 1.0); t0 = time.perf_counter(); fn(x, var[nm][0], np.random.default_rng(0))
        dt = time.perf_counter() - t0
        check(f'{nm} speed 1 s stereo', dt < 0.5, f'{dt * 1000:.0f} ms')
    x = _saw(58.11, 2.0)
    y = fx_ott(x, dict(depth=1, down_ratio=1, up_ratio=1), None)
    check('ott crossover sums flat (ratios 1)', np.abs(y - x).max() < 1e-9, f'max err {np.abs(y - x).max():.1e}')
    y = fx_mbdist(x, {}, None)
    check('mbdist sub band mono', np.abs(_lp0(y[0] - y[1], 100)).max() < 1e-3 * np.abs(y).max())
    engine.CTX.clear(); engine.CTX.update(saved)
    print('ALL PASS' if not fails else f'{fails} FAILED')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    selftest()
