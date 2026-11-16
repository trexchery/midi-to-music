"""Resampling and texture: 'resampled' (wavetable cut from a rendered preset), 'psola' (pitch-synchronous grains from a
preset or wav) and FX 'convolve' (code-generated metal/stone/glass/chain/cathedral impulse responses).

Sources are rendered once through their preset's full chain with engine.render and cached in cache/."""
import os, json, hashlib, zlib
import numpy as np, scipy.signal as ss, soundfile as sf
from scipy.ndimage import gaussian_filter1d
from numba import njit
import engine

SR = engine.SR
HERE = os.path.dirname(os.path.abspath(engine.__file__))
WT_DIR = os.path.join(HERE, 'cache', 'wavetables'); IR_DIR = os.path.join(HERE, 'cache', 'irs')
VERSION = 2                                     # bump to invalidate every cached table / IR
_MEM = {}


def _key(*a): return hashlib.sha1(json.dumps([VERSION, *a], sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cached(folder, name, build):
    """Memory, then .npy on disk, else build and save."""
    path = os.path.join(folder, name + '.npy')
    if path not in _MEM:
        if os.path.exists(path):
            _MEM[path] = np.load(path)
        else:
            os.makedirs(folder, exist_ok=True)
            _MEM[path] = build(); np.save(path, _MEM[path])
    return _MEM[path]


def _source(src, ref_midi, ref_s, tail_s, src_hz=None):
    """-> (stereo audio, its fundamental Hz, cache key). src: preset stem, inline preset dict or 'wav:<path>'."""
    if isinstance(src, str) and src.startswith('wav:'):
        path = src[4:] if os.path.isabs(src[4:]) else os.path.join(HERE, src[4:])
        def load():
            x, sr = sf.read(path, always_2d=True); x = x.T
            x = ss.resample_poly(x, SR, sr, axis=1) if sr != SR else x
            return np.vstack([x[0], x[-1]]) / max(np.abs(x).max(), 1e-9)
        k = 'wav_' + _key(path, os.path.getmtime(path)); x = _cached(WT_DIR, k, load)
        if not src_hz:                          # shortcut: one median YIN estimate; pass src_hz for unpitched or gliding files
            import librosa
            src_hz = float(np.median(librosa.yin(x.mean(0), fmin=30, fmax=1500, sr=SR, frame_length=4096)))
        return x, src_hz, k
    pre = dict(src) if isinstance(src, dict) else json.load(open(os.path.join(HERE, 'presets', src + '.json')))
    pre.setdefault('name', src if isinstance(src, str) else 'inline')
    def build():
        saved = dict(engine.CTX)                # render() overwrites CTX['notes'] / ['cents'] of the song being rendered
        try:
            return engine.render([(0.0, ref_s, ref_midi, 100)], pre, ref_s + tail_s)
        finally:
            engine.CTX.clear(); engine.CTX.update(saved)
    k = 'src_' + _key(pre, ref_midi, ref_s, tail_s)
    return _cached(WT_DIR, k, build), engine.hz(ref_midi + 12 * pre.get('octave', 0)), k


def _active(m, db=-30):
    """First/last sample whose 10 ms RMS is within db of the loudest block."""
    b = 480; r = np.sqrt((m[:len(m) // b * b].reshape(-1, b) ** 2).mean(1))
    on = np.nonzero(r > r.max() * 10 ** (db / 20))[0]
    return on[0] * b, (on[-1] + 1) * b


# ---------------------------------------------------------------- wavetable from a rendered sound

def _table(p):
    """frames x 2048 single cycles, cut period-aligned across the source render, harmonics only, unit RMS each,
    all sharing the middle frame's harmonic phases."""
    x, f_ref, k = _source(p.get('source', '08_reese_menace'), p.get('ref_midi', 46), p.get('ref_s', 2.5), p.get('ref_tail', 1.0),
                          p.get('src_hz'))
    F = p.get('frames', 128)
    def build():
        m = x.mean(0); P = SR / f_ref
        m4 = ss.resample_poly(m, 4, 1)          # 4x band-limited upsample so fractional-period reads stay clean
        a, b = _active(m)
        t0 = np.linspace(max(a, P + 1), min(b, len(m)) - P - 2, F)[:, None]
        j = np.arange(2048) / 2048
        rd = lambda s: np.interp(s * 4, np.arange(len(m4)), m4)
        fr = (1 - j) * rd(t0 + P * j) + j * rd(t0 - P + P * j)    # crossfade with the previous cycle: seamless loop
        sp = np.fft.rfft(fr, axis=1); sp[:, 0] = 0; sp[:, int(20000 / f_ref) + 1:] = 0
        sp = np.abs(sp) * np.exp(1j * np.angle(sp[F // 2]))      # one shared phase set: morphing moves only magnitudes, no pitch wobble
        fr = np.fft.irfft(sp, 2048, axis=1)
        fr /= np.sqrt((fr ** 2).mean(1, keepdims=True)) + 1e-12
        return fr / np.abs(fr).max()
    return _cached(WT_DIR, _key(k, F, f_ref), build)


@njit(cache=True)
def _wt(tab, pos, inc, ph):
    """Bilinear (frame x phase) wavetable read; inc = f/SR exactly, so pitch is exact."""
    n = pos.shape[0]; F, L = tab.shape; y = np.empty(n)
    for i in range(n):
        fp = pos[i] * (F - 1); f0 = min(int(fp), F - 2); ff = fp - f0
        x = ph * L; i0 = int(x); fr = x - i0; i1 = (i0 + 1) % L
        a = tab[f0, i0] + fr * (tab[f0, i1] - tab[f0, i0])
        b = tab[f0 + 1, i0] + fr * (tab[f0 + 1, i1] - tab[f0 + 1, i0])
        y[i] = a + ff * (b - a)
        ph += inc
        if ph >= 1.0: ph -= 1.0
    return y


# source '08_reese_menace' preset stem | inline preset dict | 'wav:<path>' (src_hz: its pitch, else estimated); ref_midi 46
# (pitch the source is rendered at); ref_s 2.5 / ref_tail 1.0 (s of note / tail rendered; the tail only matters for long
# reverbs); frames 128 (16-256 cycles in the table); order 'bright' (sort frames dark->bright) | 'time'; smooth 3 (0-10
# frames of blur between neighbours); position 0.25 (0-1 base frame); vel_pos 0.2 (0-1 velocity opens position); pos_env
# 0.45 (-1..1 extra position at note-on) decaying over pos_decay 0.5 s; pos_lfo 0.12 (0-0.5 depth) with period lfo_eighths
# 8 (eighths, note-relative); chop 0 (0-1: random position jump every eighth, 4 ms slew); unison 5 (1-9, odd keeps a voice
# on f); detune_cents 12 (0-50 outer voices); spread 0.8 (0-1 pan width); drive 0 (0-8, tanh per cycle inside the table:
# alias-free, no intermod between unison voices); cutoff 0 (Hz, 0 = off) + f_env_oct 2.5 / f_decay 0.4 s / res 0.8
# (lowpass); attack 0.01 / decay 0.8 / sustain 0.8 / release 1.2 (amp ADSR).
def v_resampled(p, f, dur, vel, rng):
    """Wavetable cut from a rendered preset, scanned by envelope + eighth-synced LFO (+ optional per-eighth chop)."""
    rel = p.get('release', 1.2); n_on = int(dur * SR); n = n_on + int(rel * SR)
    t = np.arange(n) / SR; e8 = engine.CTX['eighth']
    tab = _table(p)
    if p.get('order', 'bright') == 'bright':    # position 0 = darkest cycle, 1 = brightest
        mag = np.abs(np.fft.rfft(tab, axis=1)); tab = tab[np.argsort(mag @ np.arange(1025) / mag.sum(1))]
    if p.get('smooth', 3):                      # blur neighbouring frames (gaussian sigma in frames): glides, not zipper grit
        tab = gaussian_filter1d(tab, p.get('smooth', 3), axis=0, mode='nearest')
    u = max(1, int(p.get('unison', 5))); det = p.get('detune_cents', 12) if u > 1 else 0
    sp = np.fft.rfft(tab, axis=1)
    if p.get('drive', 0) > 0:                   # tanh on each 8192-point cycle: its new partials stay exact harmonics of f (no aliasing at any drive)
        d = p['drive']; sp = np.fft.rfft(np.tanh(d * np.fft.irfft(sp, 8192, axis=1) * 4) / np.tanh(d), axis=1) / 4
    sp[:, int(20000 / (f * 2 ** (det / 1200))) + 1:] = 0      # mip: nothing above 20 kHz
    big = np.fft.irfft(sp, 8192, axis=1) * 4    # 8192-point read table keeps linear-interpolation images below -70 dB
    pos = (p.get('position', 0.25) + p.get('vel_pos', 0.2) * vel + p.get('pos_env', 0.45) * np.exp(-t / p.get('pos_decay', 0.5))
           + p.get('pos_lfo', 0.12) * np.sin(2 * np.pi * t / (p.get('lfo_eighths', 8) * e8)))
    if p.get('chop', 0):                        # jump to a new random position every eighth (note-relative), 4 ms slew
        step = rng.uniform(-1, 1, int(t[-1] / e8) + 2)[(t / e8).astype(int)]; c = np.exp(-1 / (0.004 * SR))
        pos = pos + p['chop'] * ss.lfilter([1 - c], [1, -c], step, zi=[step[0] * c])[0]
    pos = np.clip(pos, 0, 1)
    out = np.zeros((2, n)); wsum = 0
    for o in (np.linspace(-1, 1, u) if u > 1 else [0.0]):
        w = 1 - 0.35 * abs(o)                   # centre voice loudest: odd unison keeps a voice exactly at f
        out += w * engine.pan(_wt(big, pos, f * 2 ** (o * det / 1200) / SR, rng.random()), o * p.get('spread', 0.8))
        wsum += w * w
    out /= np.sqrt(wsum) * 0.7071
    if p.get('cutoff', 0):
        fc = p['cutoff'] * 2 ** (p.get('f_env_oct', 2.5) * np.exp(-t / p.get('f_decay', 0.4)) * (0.5 + 0.5 * vel))
        out = np.stack([engine.svf(c, fc, p.get('res', 0.8), 0, SR) for c in out])
    return out * engine.adsr(n_on, n, p.get('attack', 0.01), p.get('decay', 0.8), p.get('sustain', 0.8), rel)


# ---------------------------------------------------------------- pitch-synchronous granular

@njit(cache=True)
def _psola(src4, n, P, L, pos):
    """Hann grains of length L read from src4 (4x upsampled) at source sample pos[k], placed every P samples (fractional)."""
    out = np.zeros(n); m = src4.shape[0]
    for k in range(pos.shape[0]):
        tk = k * P
        for i in range(int(np.ceil(tk)), min(n, int(tk + L) + 1)):
            u = (i - tk) / L
            if u >= 1.0: break
            s = pos[k] * 4 + (i - tk) * 4; j = int(s); t = s - j
            if j >= 1 and j + 2 < m:            # 4-point Hermite: interpolation error stays below the -60 dB floor
                y0, y1, y2, y3 = src4[j - 1], src4[j], src4[j + 1], src4[j + 2]
                c1 = 0.5 * (y2 - y0); c2 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3; c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
                out[i] += (0.5 - 0.5 * np.cos(2 * np.pi * u)) * (((c3 * t + c2) * t + c1) * t + y1)
    return out


def _marks(m, f0):
    """Unwrapped phase of the source's fundamental (analytic Gaussian band around f0), made non-decreasing."""
    sp = np.fft.fft(m); fq = np.fft.fftfreq(len(m), 1 / SR)
    ph = np.fft.ifft(sp * np.exp(-0.5 * ((fq - f0) / (0.25 * f0)) ** 2) * (fq > 0))
    return np.maximum.accumulate(np.unwrap(np.angle(ph)))


def _lock(ph, s):
    """Pitch-marked read positions: grain k starts at the point nearest its scan target s[k] where the source's
    fundamental has the same phase as at grain 0. Every grain then adds in phase, so the output is exactly periodic
    at the placement rate even when the source drifts or has vibrato (no PSOLA phase drift = no pitch error)."""
    i = np.arange(len(ph), dtype=float); q = np.interp(s, i, ph)
    return np.interp(q[0] + 2 * np.pi * np.round((q - q[0]) / (2 * np.pi)), ph, i)


# source '03_dark_tines' preset stem | inline preset dict | 'wav:<path>' (src_hz: its pitch, else estimated);
# ref_midi 58 / ref_s 3.0 / ref_tail 1.0 (source render); grain_periods 2.0 (1-8: grain length in periods of f, longer
# = more of the source's own pitch and movement smeared in); scan_speed 0.3 (-2..2, 1 = source realtime, 0 = freeze,
# ping-pongs at the ends); start 0.2 (0-1 position in the source); spray 0 (0-60 ms random read jitter: breath and
# grit between the harmonics); stereo_ms 37 (0-200 right scan offset); attack 0.06 / decay 1.0 / sustain 0.85 /
# release 1.5 (amp ADSR).
def v_psola(p, f, dur, vel, rng):
    """Grains of the source placed exactly every 1/f s: pitch = f, texture = source. Pitch-marked grain reads (_lock)
    make a periodic source purely harmonic; scanning a detuned, chorused or noisy source adds a unison-like cluster
    around each harmonic (scan_speed 0 freezes it back to exact harmonics)."""
    rel = p.get('release', 1.5); n_on = int(dur * SR); n = n_on + int(rel * SR)
    src = p.get('source', '03_dark_tines')
    x, f_src, k = _source(src, p.get('ref_midi', 58), p.get('ref_s', 3.0), p.get('ref_tail', 1.0), p.get('src_hz'))
    if k + '_x4' not in _MEM:
        _MEM[k + '_x4'] = ss.resample_poly(x, 4, 1, axis=1)
    x4 = _MEM[k + '_x4']
    if (k, f_src) not in _MEM:
        _MEM[k, f_src] = [_marks(c, f_src) for c in x]
    P = SR / f; gp = p.get('grain_periods', 2.0); L = gp * P; Ps = SR / f_src
    a, b = _active(x.mean(0), -20)              # scan only the loud part of the source, before a preset's release
    b = b if str(src).startswith('wav:') else min(b, int(p.get('ref_s', 3.0) * SR))
    span = max(1.0, min(b, x.shape[1] - L - Ps - 2) - a)
    tk = np.arange(int(n / P) + 2) * P
    out = np.zeros((2, n))
    for ch in (0, 1):
        scan = (p.get('start', 0.2) * span + p.get('scan_speed', 0.3) * tk + ch * p.get('stereo_ms', 37) / 1000 * SR
                + rng.normal(0, p.get('spray', 0) / 1000 * SR, tk.shape))
        r = scan % (2 * span)
        out[ch] = _psola(x4[ch], n, P, L, _lock(_MEM[k, f_src][ch], a + np.where(r < span, r, 2 * span - r)))   # ping-pong scan
    out /= max(1.0, gp / 2)                     # Hann overlap-add gain
    return out * engine.adsr(n_on, n, p.get('attack', 0.06), p.get('decay', 1.0), p.get('sustain', 0.85), rel)


VOICES = {'resampled': v_resampled, 'psola': v_psola}


# ---------------------------------------------------------------- convolution with generated impulse responses

IR_DEF = {'metal': 4.0, 'stone': 0.8, 'glass': 2.5, 'chain': 1.6, 'cathedral': 7.0}    # default decay_s (T60)


def _modes(n, fr, amp, t60, r, ph=None):
    """Sum of exponentially decaying sines (one row per mode) into n samples."""
    t = np.arange(n) / SR; y = np.zeros(n)
    ph = r.uniform(0, 6.28, len(fr)) if ph is None else ph
    for f_, a_, d_, p_ in zip(fr, amp, t60, ph):
        k = min(n, int(d_ * SR * 1.2)) if d_ > 0 else n
        y[:k] += a_ * np.exp(-6.9 * t[:k] / d_) * np.sin(2 * np.pi * f_ * t[:k] + p_)
    return y


def _ir(kind, T, br):
    r = np.random.default_rng(zlib.crc32(kind.encode()))
    n = int(T * 1.2 * SR) + 1; t = np.arange(n) / SR; out = np.zeros((2, n))
    if kind == 'metal':                         # plate/bar: dense inharmonic partials, highs die first
        fr = np.concatenate([110 * np.array([1, 2.756, 5.404, 8.933, 13.34]), np.exp(r.uniform(np.log(70), np.log(11000), 180))])
        amp = (fr / 500) ** (-1.3 + 1.0 * br) * r.uniform(0.3, 1, len(fr)); amp[:5] *= 3
        t60 = T * np.clip((fr / 400) ** (-0.5 * (1.2 - br)), 0.08, 1)
        for c in (0, 1): out[c] = _modes(n, fr * (1 + r.uniform(-1e-3, 1e-3, len(fr))), amp, t60, r)
    elif kind == 'stone':                       # dense low body + crumbly noise
        fr = np.exp(r.uniform(np.log(40), np.log(900), 260)); amp = (fr / 100) ** -0.8 * r.uniform(0.2, 1, len(fr))
        t60 = T * r.uniform(0.3, 1, len(fr))
        for c in (0, 1):
            nz = engine.svf(r.standard_normal(n) * np.exp(-6.9 * t / (T * 0.5)), np.full(n, 600 + 6000 * br), 0.7, 0, SR)
            out[c] = _modes(n, fr, amp, t60, r) / 30 + nz
    elif kind == 'glass':                       # sparse high partials in close pairs that beat
        base = np.exp(r.uniform(np.log(700), np.log(9000), 28))
        fr = np.concatenate([base, base + r.uniform(0.4, 3, len(base))]); amp = (fr / 2000) ** (-1.5 + br) * r.uniform(0.3, 1, len(fr))
        t60 = T * np.clip((fr / 1500) ** -0.6, 0.15, 1)
        for c in (0, 1): out[c] = _modes(n, fr, amp, t60, r)
    elif kind == 'chain':                       # rattle: a thinning stream of tiny metallic clicks
        rate = 220 * np.exp(-t / (T / 3)); hits = np.nonzero(r.random(n) < rate / SR)[0]
        k = int(0.06 * SR); fr = r.uniform(1500, 9000, 6)
        for h in hits:
            c = r.integers(2); m = min(k, n - h)
            out[c, h:h + m] += np.exp(-6.9 * h / SR / T) * r.uniform(0.3, 1) * _modes(k, fr * r.uniform(0.9, 1.1, 6), np.ones(6),
                                                                                     r.uniform(0.01, 0.04, 6) * (0.5 + br), r)[:m]
    elif kind == 'cathedral':                   # diffuse noise tail whose lowpass sinks over time (air absorption)
        fc = 300 + (1500 + 13000 * br) * np.exp(-t / (T * 0.25))
        er = np.zeros(n); er[r.integers(int(0.008 * SR), int(0.09 * SR), 14)] = r.uniform(-1, 1, 14)
        for c in (0, 1):
            nz = r.standard_normal(n) * np.exp(-6.9 * t / T) * np.clip(t / 0.03, 0, 1)
            out[c] = engine.svf(engine.svf(nz + er, fc, 0.6, 0, SR), fc, 0.6, 0, SR)
    else:
        raise ValueError(f'convolve: unknown kind {kind!r}; have {list(IR_DEF)}')
    return out / np.sqrt((out ** 2).sum() / 2)  # unit energy per channel on average


# kind 'metal' | 'stone' | 'glass' | 'chain' | 'cathedral'; decay_s (T60 s, default per kind: 4 / 0.8 / 2.5 / 1.6 / 7);
# brightness 0.5 (0-1 high-frequency level and decay); predelay_ms 20 (0-500); lowcut 120 (Hz on the wet, 0 = off);
# mix 0.35 (0-1, wet is RMS-matched to dry first).
def fx_convolve(x, p, rng):
    """Convolution with a generated IR. LTI: never changes pitch; wet is RMS-matched to dry, then mixed.
    metal/glass/chain rings at its own inharmonic modes after transients (that is the point; mix sets how much)."""
    kind = p.get('kind', 'metal'); T = p.get('decay_s', IR_DEF.get(kind, 2.0)); br = p.get('brightness', 0.5)
    ir = _cached(IR_DIR, f'{kind}_{_key(kind, T, br)}', lambda: _ir(kind, T, br))
    ir = np.pad(ir, ((0, 0), (int(p.get('predelay_ms', 20) / 1000 * SR), 0)))
    wet = ss.fftconvolve(x, ir, axes=1)[:, :x.shape[1]]
    if p.get('lowcut', 120):                    # keep the sub out of long tails
        wet = ss.sosfilt(ss.butter(2, p.get('lowcut', 120), 'highpass', fs=SR, output='sos'), wet)
    wet *= np.sqrt((x ** 2).mean() / max((wet ** 2).mean(), 1e-20))
    mix = p.get('mix', 0.35)
    return (1 - mix) * x + mix * wet


FX = {'convolve': fx_convolve}


# ---------------------------------------------------------------- self-test

def selftest():
    import sys, time
    bad = []
    def check(name, ok, info=''):
        print(f'{"PASS" if ok else "FAIL"} {name} {info}'); ok or bad.append(name)
    def spec(x, seg):
        m = x.mean(0)[seg]; N = 1 << int(np.ceil(np.log2(len(m)))) + 3
        return np.abs(np.fft.rfft(m * ss.windows.blackmanharris(len(m)), N)), SR / N
    def pitch(x, f, seg, strongest=False):
        if strongest:                           # strongest of harmonics 1-8, compared with its own k*f
            s, df = spec(x, seg); h = 1 + int(np.argmax([s[int(k * f * 0.99 / df):int(k * f * 1.01 / df)].max() for k in range(1, 9)]))
            return pitch(x, h * f, seg)
        s, df = spec(x, seg); lo, hi = int(f * 0.97 / df), int(f * 1.03 / df)
        i = lo + np.argmax(s[lo:hi]); a, b, c = np.log(s[i - 1:i + 2] + 1e-30)
        return 1200 * np.log2((i + 0.5 * (a - c) / (a - 2 * b + c)) * df / f)
    def alias(x, f, seg):
        s, df = spec(x, seg); fq = np.arange(len(s)) * df; P_ = s ** 2
        far = np.abs(fq - np.round(fq / f) * f) > 0.02 * f
        return 10 * np.log10(P_[far & (fq > 15)].sum() / P_.sum() + 1e-30)
    F3 = (308.87, 462.78, 58.11); seg = slice(int(0.8 * SR), int(3.0 * SR))
    periodic = {'name': 'test_fm', 'engine': 'fm', 'voice': {'pairs': [[1, 1, 2.5, 9, 1, 9], [1, 2, 1.0, 9, 0.5, 9]], 'spread_cents': 0}, 'fx': []}
    vp = {'resampled': {'unison': 1}, 'psola': {}}
    # aliasing = nonlinear / interpolation products; position motion and spray add (non-alias) sidebands, measured apart
    agg = {'resampled': {'unison': 1, 'drive': 8, 'position': 1.0, 'pos_env': 0, 'pos_lfo': 0, 'cutoff': 0},
           'psola': {'source': periodic, 'grain_periods': 1.0, 'scan_speed': 1.0, 'spray': 0}}
    for name, fn in VOICES.items():
        t = time.time(); fn(vp[name], 100.0, 0.5, 0.8, np.random.default_rng(0))
        check(f'{name} build+first note', True, f'{time.time() - t:.2f}s (source render + cache)')
        for f in F3:
            y = fn(vp[name], f, 3.2, 0.8, np.random.default_rng(1))
            ok = y.shape == (2, int(3.2 * SR) + int(vp[name].get('release', 1.2 if name == 'resampled' else 1.5) * SR)) and np.isfinite(y).all()
            check(f'{name} shape/finite {f}', ok, f'{y.shape} peak {np.abs(y).max():.2f}')
            c = pitch(y, f, seg, name == 'psola'); check(f'{name} pitch {f} Hz', abs(c) < 1, f'{c:+.3f} c')
        y = fn(agg[name], 462.78, 3.2, 1.0, np.random.default_rng(2)); a = alias(y, 462.78, seg)
        check(f'{name} alias 462.78 aggressive', a < -60, f'{a:.1f} dB')
        y1, y2 = (fn(vp[name], 200.0, 1.0, 0.7, np.random.default_rng(5)) for _ in range(2))
        check(f'{name} deterministic', np.array_equal(y1, y2))
        t = time.time(); fn(vp[name], 200.0, 1.0, 0.7, np.random.default_rng(6)); dt = time.time() - t
        check(f'{name} speed 1 s note', dt < 0.5, f'{dt:.3f}s')
    y = v_resampled({'unison': 5, 'chop': 0.6}, 308.87, 3.2, 0.8, np.random.default_rng(3))
    check('resampled chop+unison finite', np.isfinite(y).all() and np.abs(y).max() < 1.5, f'peak {np.abs(y).max():.2f}')
    y = v_resampled(agg['resampled'], 1850.0, 3.2, 1.0, np.random.default_rng(2)); a = alias(y, 1850.0, seg)
    check('resampled alias 1850 drive 8', a < -60, f'{a:.1f} dB')
    for f in (155.4, 462.78):                   # inharmonic modal source + scan: pitch marks keep every grain in phase
        c = pitch(v_psola({'source': '05_bowed_steel'}, f, 3.2, 0.8, np.random.default_rng(1)), f, seg, True)
        check(f'psola bowed-steel source pitch {f} Hz', abs(c) < 1, f'{c:+.3f} c')
    for kind in IR_DEF:
        T = IR_DEF[kind]; nT = int((T * 1.2 + 0.05) * SR); ok_p = []; t = time.time()
        for f in F3:
            n = nT + int(4.2 * SR); tt = np.arange(n) / SR
            x = engine.pan(sum(np.sin(2 * np.pi * h * f * tt) / h for h in range(1, 9) if h * f < 18000), 0) * 0.3
            y = fx_convolve(x, {'kind': kind, 'mix': 1.0}, np.random.default_rng(0))
            sg = slice(nT, nT + int(4 * SR))
            ok_p.append(max(abs(pitch(y, h * f, sg) - pitch(x, h * f, sg)) for h in (1, 2, 3)))
            if f == 462.78: a = alias(y, f, sg)
        check(f'convolve {kind} shape/finite', y.shape == x.shape and np.isfinite(y).all(), f'{time.time() - t:.1f}s incl. IR build')
        check(f'convolve {kind} harmonic peaks unchanged', max(ok_p) < 1, f'max {max(ok_p):.3f} c')
        check(f'convolve {kind} no new partials in steady state 462.78', a < -60, f'{a:.1f} dB')
    x = np.random.default_rng(9).standard_normal((2, SR * 4)) * 0.1
    y1, y2 = (fx_convolve(x, {'kind': 'cathedral'}, np.random.default_rng(k)) for k in (1, 2))
    check('convolve deterministic', np.array_equal(y1, y2))
    t = time.time(); fx_convolve(x, {'kind': 'metal'}, None); check('convolve speed 4 s stereo', time.time() - t < 2, f'{time.time() - t:.2f}s')
    print('ALL PASS' if not bad else f'{len(bad)} FAILED: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    selftest()
