"""Note-aware effects: processing re-tuned to whatever MIDI note is sounding (engine.CTX['notes']).

comb     tuned feedback comb / resonator bank (string-like ring, hollow odd-harmonic tube with negative feedback)
formant  harmonic-snapped vowel morph: the "talking" / "yoy" growl, every formant sits on a harmonic of the note
sieve    STFT harmonic mask: keeps only energy near the note's harmonics, turns grit and noise into an in-tune tone
fshift   single-sideband frequency shifter: by k*f (stays harmonic) or by a few Hz (detune), optional echo spiral
freeze   spectral freeze: captures each note (or eighth) and holds it as a phase-vocoder pad smear

All of them follow the most recent note at each sample (equal starts: the lowest pitch); before the first note
the input passes through dry. Every effect only filters or moves energy onto harmonics of the current note, except
fshift mode 'hz' (detunes on purpose) and comb 'glide' (a moving delay adds Doppler while it sweeps).
"""
import numpy as np, scipy.signal as ss, scipy.fft as sf
from numba import njit
import engine

SR = engine.SR

# bass-singer formants (Hz, dB), F1..F4
VOWELS = {'a': ((600, 1040, 2250, 2450), (0, -7, -9, -9)),
          'e': ((400, 1620, 2400, 2800), (0, -12, -9, -12)),
          'i': ((250, 1750, 2600, 3050), (0, -30, -16, -22)),
          'o': ((400, 750, 2400, 2600), (0, -11, -21, -20)),
          'u': ((350, 600, 2400, 2675), (0, -20, -32, -28))}


# ---------------------------------------------------------------- shared helpers

def _track(n):
    """Per-sample Hz of the most recent note (0 before the first), position 0..1 through it, and its segments."""
    f = np.zeros(n); pos = np.zeros(n); segs = []
    notes = sorted(engine.CTX['notes'], key=lambda q: (q[0], -q[2]))     # equal starts: the lowest pitch wins
    for j, (st, du, m, v) in enumerate(notes):
        i0 = max(0, int(round(st * SR)))
        i1 = min(n, int(round(notes[j + 1][0] * SR))) if j + 1 < len(notes) else n
        if i1 <= i0: continue
        f[i0:i1] = engine.hz(m)
        pos[i0:i1] = np.clip((np.arange(i0, i1) / SR - st) / max(du, 1e-3), 0, 1)
        segs.append((i0, i1, engine.hz(m), st, du))
    return f, pos, segs


def _sweep(n):
    """0 -> 1 over one eighth, back over the next: a smooth per-eighth trajectory locked to the MIDI grid."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / SR / engine.CTX['eighth'])


def _mix(x, wet, mix, segs, match=True):
    """Dry before the first note; after it blend in the wet (RMS-matched to the dry so mix means the same everywhere)."""
    if not segs: return x
    i0 = segs[0][0]; m = np.zeros(x.shape[1]); m[i0:] = mix
    L = min(240, x.shape[1] - i0); m[i0:i0 + L] *= np.linspace(0, 1, L)
    if match:
        wet = wet * np.sqrt((x[:, i0:] ** 2).mean() / max((wet[:, i0:] ** 2).mean(), 1e-30))
    return x * (1 - m) + wet * m


def _drive(x, d):
    """tanh saturation at 4x oversampling (harmonics of the note, no aliasing)."""
    if d <= 1: return x
    up = ss.resample_poly(x, 4, 1, axis=1, window=('kaiser', 10.0))
    return ss.resample_poly(np.tanh(d * up) / np.tanh(d), 1, 4, axis=1, window=('kaiser', 10.0))[:, :x.shape[1]]


# ---------------------------------------------------------------- comb

@njit(cache=True)
def _comb(x, d, fb, h):
    """y = x + fb * (h * y)(i - d[i]): symmetric (linear-phase) loop filter h centred on a cubic-Lagrange fractional tap."""
    n = x.shape[0]; y = np.zeros(n); M = h.shape[0] // 2
    for i in range(n):
        j = i - d[i]; i0 = int(np.floor(j)); fr = j - i0; v = 0.0
        c0 = -fr * (fr - 1) * (fr - 2) / 6; c1 = (fr + 1) * (fr - 1) * (fr - 2) / 2
        c2 = -(fr + 1) * fr * (fr - 2) / 2; c3 = (fr + 1) * fr * (fr - 1) / 6
        for m in range(-M, M + 1):
            q = i0 + m - 1
            if q >= 0:
                v += h[m + M] * (c0 * y[q] + c1 * y[q + 1] + c2 * y[q + 2] + c3 * y[q + 3])
            elif q + 3 >= 0:
                for t in range(4):
                    if q + t >= 0: v += h[m + M] * (c0, c1, c2, c3)[t] * y[q + t]
        y[i] = x[i] + fb * v
    return y


def fx_comb(x, p, rng):
    """Feedback comb tuned to the note (retuned at each note start, the old ring fades over ring_ms)."""
    n = x.shape[1]; f, pos, segs = _track(n)
    fb = float(np.clip(p.get('feedback', 0.9), -0.995, 0.995))  # -0.995..0.995; <0 = odd harmonics only (hollow), still at the note
    dm = p.get('damping', 0.3)                      # 0..1 loop lowpass: higher = darker, highs die sooner
    hk = (1 - dm) * np.eye(7)[3] + dm * np.hanning(9)[1:-1] / np.hanning(9).sum()  # linear phase: damping cannot detune any harmonic
    k = p.get('k', 1)                               # harmonic the comb rings at: 1 = note, 2 = octave, 3 = 12th; 0.5 = octave down
    X = max(1, int(p.get('ring_ms', 50) / 1000 * SR))   # 5..2000 ms the previous note's ring survives a note change
    ks = [k * r for r in p.get('ratios', [0.5, 1, 2, 3])] if p.get('mode', 'comb') == 'bank' else [k]  # bank: octaves + 12th
    glide = p.get('glide')                          # None | 'note' | 'eighth': flanger sweep k -> k2 (Doppler while moving)
    res = np.zeros_like(x); F = 120; top = SR / 6 / (2 if fb < 0 else 1)   # top: highest ring the 6-sample minimum loop allows
    for i0, i1, fn, st, du in segs:
        b = min(n, i1 + max(X, F)); q = np.arange(i0, b); t = q / SR
        s = 0.0 if not glide else (np.clip((t - st) / max(du, 1e-3), 0, 1) if glide == 'note' else _sweep(b)[i0:])
        g = np.ones(b - i0); g[i1 - i0:] = 0.5 + 0.5 * np.cos(np.pi * np.minimum(np.arange(b - i1) / X, 1))
        # 2.5 ms complementary input crossfade between consecutive note combs: no input step (click) into either loop
        win = np.sin(0.5 * np.pi * np.clip((q - i0 + 0.5) / F, 0, 1)) ** 2 * np.cos(0.5 * np.pi * np.clip((q - i1 + 0.5) / F, 0, 1)) ** 2
        for kb in ks:
            kk = kb * (1 + (p.get('k2', 2) / k - 1) * s)        # every ratio glides by the same k2/k factor
            kk = np.where(fn * kk > top, max(1, np.floor(top / fn)), kk)  # too high to ring: highest harmonic that fits (still in tune)
            d = np.maximum(6.0, np.broadcast_to(SR / (fn * kk) / (2 if fb < 0 else 1), (b - i0,)))  # fb<0 rings at odd multiples of 1/(2d)
            for ch in range(x.shape[0]):
                xin = x[ch, i0:b] * win                           # the old comb only rings out, the next note feeds the new one
                res[ch, i0:b] += g * (_comb(xin, np.ascontiguousarray(d, dtype=np.float64), fb, hk) - xin)
    # positive feedback also rings at 0 Hz; an in-loop DC blocker would detune the upper harmonics, so cut it after the loop
    wet = ss.sosfilt(ss.butter(4, 20, 'highpass', fs=SR, output='sos'), x + res / len(ks))
    return _mix(x, wet, p.get('mix', 0.5), segs)                # mix 0..1


# ---------------------------------------------------------------- formant

def fx_formant(x, p, rng):
    """Vowel morph through parallel bandpasses on the two harmonics of the current note around each formant (crossfaded)."""
    n = x.shape[1]; f, pos, segs = _track(n)
    vs = list(p.get('vowels', ['u', 'o', 'a', 'e', 'i']))       # path through VOWELS keys
    vs = vs * 2 if len(vs) == 1 else vs
    s = (pos if p.get('morph', 'note') == 'note' else _sweep(n)) * (len(vs) - 1)  # 'note': start->end over each note; 'eighth'
    j = np.minimum(s.astype(int), len(vs) - 2); fr = (s - j)[:, None]
    F = np.log([VOWELS[v][0] for v in vs]); G = np.array([VOWELS[v][1] for v in vs], float)
    fc = np.exp(F[j] * (1 - fr) + F[j + 1] * fr)
    gdb = (G[j] * (1 - fr) + G[j + 1] * fr) * p.get('gain_depth', 0.5)  # 0..1 how much of the vowel's formant levels to keep
    fn = np.where(f > 0, f, fc[:, 0])[:, None]                  # before the first note (dry there): any grid
    h = np.maximum(fc / fn, 1); lo = np.floor(h); u = h - lo
    c = np.exp(-1 / (max(p.get('glide_ms', 25), 0.1) / 1000 * SR))  # 5..100 ms smoothing of centres and levels at note changes / vowel jumps
    sm = lambda a: ss.lfilter([1 - c], [1, -c], a, axis=0, zi=c * a[:1])[0]
    q = p.get('q', 6.0)                                         # 2..16 per pass (two passes): higher = more vocal / whistling
    xd = _drive(x, p.get('drive', 2.0))                         # 1..10 tanh drive before the filters (1 = clean)
    wet = np.zeros_like(x)
    for par in (0, 1):
        # each formant = a bandpass on its even and one on its odd neighbouring harmonic, equal-power crossfaded by where the
        # vowel target sits between them: every filter stays on a harmonic and only hops while its weight is 0 (no level dips)
        own = lo % 2 == par
        fcp = sm(np.where(own, lo, lo + 1) * fn)
        gp = sm(np.where(own, np.cos(0.5 * np.pi * u), np.sin(0.5 * np.pi * u)) * 10 ** (gdb / 20)) / q ** 2  # svf bp peaks at q per pass
        for i in range(int(np.clip(p.get('formants', 4), 1, 4))):  # 3..4
            fi = np.ascontiguousarray(fcp[:, i]); fr_ = np.ascontiguousarray(fi[::-1])
            for ch in range(x.shape[0]):                        # forward + backward pass: zero phase, so a moving formant cannot bend pitch
                y = engine.svf(np.ascontiguousarray(xd[ch]), fi, q, 1, SR)
                wet[ch] += gp[:, i] * engine.svf(np.ascontiguousarray(y[::-1]), fr_, q, 1, SR)[::-1]
    return _mix(x, wet, p.get('mix', 0.8), segs)                # mix 0..1


# ---------------------------------------------------------------- sieve

def _sieve_mask(fr, f, width, tilt, hmax):
    h = np.clip(np.round(fr / f), 1, hmax)
    sig = np.maximum(h * f * (2 ** (width / 1200) - 1), fr[1])  # at least one bin wide, so the pass band centres exactly on h*f
    g = np.exp(-0.5 * ((fr - h * f) / sig) ** 2) * 10 ** (tilt * np.log2(h) / 20)
    g[fr > (hmax + 0.5) * f] = 0
    return g


def fx_sieve(x, p, rng):
    """Keeps only energy near harmonics of the note sounding at each STFT frame."""
    n = x.shape[1]; f, pos, segs = _track(n)
    N = p.get('nfft', 8192); H = N // 8                         # 4096..16384: longer = narrower, purer, slower to follow
    fr, t, Z = ss.stft(np.pad(x, ((0, 0), (0, max(0, N - n)))), SR, 'hann', N, N - H)   # pad: buffers shorter than one window
    ff = f[np.clip(np.round(t * SR).astype(int), 0, n - 1)]     # the note at each frame's centre
    M = np.ones((len(fr), len(t)))
    for fn in np.unique(ff[ff > 0]):
        M[:, ff == fn] = _sieve_mask(fr, fn, p.get('width_cents', 15),  # 3..100 cents: narrow = pure tone, wide = keeps grit
                                     p.get('tilt', -3.0),               # -12..+6 dB per octave of harmonic number
                                     max(1, p.get('max_harmonic', 40)))[:, None]  # 1..200
    wet = ss.istft(Z * M, SR, 'hann', N, N - H)[1][:, :n]
    wet = np.pad(wet, ((0, 0), (0, n - wet.shape[1])))
    return _mix(x, wet, p.get('mix', 0.8), segs)                # mix 0..1


# ---------------------------------------------------------------- fshift

def _ssb(x, shift):
    """Single-sideband shift by a per-sample Hz curve; the analytic signal is band-limited so nothing passes Nyquist."""
    n = x.shape[1]; nf = sf.next_fast_len(n)
    fr = sf.fftfreq(nf, 1 / SR)
    h = ((fr > 0) & (fr < 0.47 * SR - max(shift.max(), 0))) * 2.0; h[0] = 1
    a = sf.ifft(sf.fft(x, nf, axis=1, workers=-1) * h, axis=1, workers=-1)[:, :n]
    return np.real(a * np.exp(2j * np.pi * np.cumsum(shift) / SR))


def fx_fshift(x, p, rng):
    """Frequency shift by k*f of the current note (harmonic) or by a fixed Hz (detune), with an optional echo spiral."""
    n = x.shape[1]; f, pos, segs = _track(n)
    if p.get('mode', 'harmonic') == 'harmonic':
        sh = p.get('k', 1) * f                      # integer k (-8..8): every partial lands on a harmonic; -1 folds down
        w = np.hanning(242)[1:-1]; sh = np.convolve(np.pad(sh, 120, 'edge'), w / w.sum(), 'valid')[:n]  # 5 ms glide at note changes: a shift step clicks
    else:
        sh = np.full(n, float(p.get('hz', 1.5)))    # 'hz' mode: -20..20 Hz, inharmonic by design (detune / beating tails)
    wet = _ssb(x, sh)
    fb = float(np.clip(p.get('feedback', 0.0), -0.95, 0.95))  # 0..0.9 spiral: each echo shifted again (climbs the harmonic series)
    D = max(1, int(round(p.get('delay', 1) * engine.CTX['eighth'] * SR)))  # echo spacing in eighths (0.25..8)
    echo = wet; m = 1
    while fb and m * D < n and abs(fb) ** m > 1e-3:
        src = np.zeros_like(echo); src[:, D:] = echo[:, :-D]
        shm = np.zeros(n); shm[m * D:] = sh[:n - m * D]            # shifted by the note it came from, so it stays harmonic
        echo = fb * _ssb(src, shm); wet = wet + echo; m += 1
    wet = ss.sosfilt(ss.butter(2, 20, 'highpass', fs=SR, output='sos'), wet)   # a partial shifted onto 0 Hz is DC
    return _mix(x, wet, p.get('mix', 0.5), segs, match=False)   # mix 0..1


# ---------------------------------------------------------------- freeze

def fx_freeze(x, p, rng):
    """Captures the spectrum after each note start (or eighth) and holds it as a phase-vocoder sustain."""
    n = x.shape[1]; f, pos, segs = _track(n)
    if not segs: return x
    N = 4096; H = N // 4; ha = 256; w = ss.get_window('hann', N); kb = np.arange(N // 2 + 1)
    off = p.get('capture_offset_ms', 60) / 1000     # 0..500 ms after the note start where the capture window begins: skip the attack, catch the body
    hold = p.get('hold_s', 2.0)                     # 0.2..10 s each capture rings
    dif = p.get('diffusion', 1.0)                   # 0..10 cents slow drift per partial (capped at 2 Hz): 0 = glassy, 4+ = chorused smear
    xf = max(p.get('crossfade', 0.15), 1e-3)        # 0.01..2 s fade in/out of each held spectrum
    if p.get('every', 'note') == 'note':            # 'note' | 'eighth'
        caps = [st for *_, st, _ in segs]
    else:
        e = engine.CTX['eighth']; g = np.arange(np.ceil(segs[0][0] / SR / e), n / SR / e) * e
        caps = [c for c in g if f[min(int(c * SR), n - 1)] > 0 and pos[min(int(c * SR), n - 1)] < 1]   # only while a note is held
    M = max(2, int(hold * SR / H)); tm = np.arange(M) * H / SR
    env = np.clip(tm / xf, 0, 1) * np.clip((hold - tm) / xf, 0, 1)
    xp = np.pad(x, ((0, 0), (N, N + M * H + N))); out = np.zeros_like(xp)
    for t0 in caps:
        c = max(int(round((t0 + off) * SR)), 0) + N + N // 2       # analysis window starts at the capture time: no pre-note audio (detunes)
        if c >= N + n: continue
        for ch in range(x.shape[0]):
            X1 = np.fft.rfft(w * xp[ch, c - N // 2:c + N // 2]); X2 = np.fft.rfft(w * xp[ch, c - N // 2 + ha:c + N // 2 + ha])
            mag = np.abs(X2); dph = np.angle(X2) - np.angle(X1) - 2 * np.pi * kb * ha / N
            om = 2 * np.pi * kb / N + ((dph + np.pi) % (2 * np.pi) - np.pi) / ha    # true frequency per bin (rad/sample)
            pk = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
            pk = pk if pk.size else np.array([int(mag.argmax())])
            own = pk[np.searchsorted((pk[:-1] + pk[1:]) / 2, kb)]   # identity phase locking: each bin follows its peak
            r = rng.uniform(1.0, 3.0, kb.size)[own]; th = rng.uniform(0, 2 * np.pi, kb.size)[own]
            # PM index for +-dif cents; capped at 2 Hz, beyond that a phase-only change between overlapping frames smears into sidebands
            A = np.minimum(dif / 1200 * np.log(2) * np.abs(om[own]) * SR / (2 * np.pi), 2.0) / r
            # zero-mean slow vibrato per partial (1-3 Hz): a chorus around the exact pitch, sidebands within a few Hz
            ph = np.angle(X2) + om[own] * (np.arange(M) * H)[:, None] + A * (np.sin(2 * np.pi * r * tm[:, None] + th) - np.sin(th))
            fr = np.fft.irfft(mag * np.exp(1j * ph), N, axis=1) * w * (env / 1.5)[:, None]   # hann^2 at N/4 hop sums to 1.5
            s0 = c + ha - N // 2
            for m in range(M): out[ch, s0 + m * H:s0 + m * H + N] += fr[m]
    return _mix(x, out[:, N:N + n], p.get('mix', 0.5), segs)    # mix 0..1


FX = {'comb': fx_comb, 'formant': fx_formant, 'sieve': fx_sieve, 'fshift': fx_fshift, 'freeze': fx_freeze}


# ---------------------------------------------------------------- selftest

def selftest():
    import sys, time
    fails = []

    def check(name, ok, info=''):
        print(f"{'PASS' if ok else 'FAIL'} {name} {info}"); fails.append(name) if not ok else None

    def note(f, dur=2.5, st=0.0):
        engine.CTX['notes'] = [(st, dur, 69 + 12 * np.log2(f / 440), 100)]

    def tone(f, dur=2.5, ramp=0.01):
        t = np.arange(int(dur * SR)) / SR
        s = sum(np.sin(2 * np.pi * h * f * t + h) / h for h in range(1, int(15000 / f) + 1)) * (0.5 - 0.5 * np.cos(np.pi * np.minimum(1, t / ramp)))
        s *= 0.5 / np.abs(s).max(); return np.stack([s, 0.9 * s])

    def peaks_cents(y, f, a=0.6, b=1.6, top=3, harmonic_input=True):
        """Worst cent error of the strongest peaks vs the nearest harmonic of f (harmonic input: the 3 strongest harmonics)."""
        seg = y[int(a * SR):int(b * SR)]; L = len(seg) * 16
        S = np.log(np.abs(np.fft.rfft(seg * ss.windows.blackmanharris(len(seg)), L)) + 1e-12); fr = np.arange(S.size) * SR / L
        if harmonic_input:      # the max inside +-1.5% of each harmonic, so slow-modulation sidebands are never mistaken for a partial
            hs = np.arange(1, int(15000 / f) + 1)
            pk = np.array([np.flatnonzero(np.abs(fr - h * f) < 0.015 * h * f)[0] + S[np.abs(fr - h * f) < 0.015 * h * f].argmax() for h in hs])
        else:
            pk = ss.argrelmax(S)[0]
        pk = pk[np.argsort(S[pk])[-top:]]
        worst = 0.0
        for i in pk:
            d = 0.5 * (S[i - 1] - S[i + 1]) / (S[i - 1] - 2 * S[i] + S[i + 1])
            pf = (i + d) * SR / L; h = max(1, round(pf / f))
            worst = max(worst, abs(1200 * np.log2(pf / (h * f))))
        return worst

    def inharm_db(y, f, a=0.6, b=1.6):
        seg = y[int(a * SR):int(b * SR)]
        S = np.abs(np.fft.rfft(seg * ss.windows.blackmanharris(len(seg)))) ** 2; fr = np.fft.rfftfreq(len(seg), 1 / SR)
        h = np.round(fr / f); bad = (fr > 20) & ((h < 1) | (np.abs(fr - h * f) > 0.02 * np.maximum(h, 1) * f))
        return 10 * np.log10(S[bad].sum() / S.sum() + 1e-30)

    run = lambda kind, x, p, seed=0: FX[kind](x, dict(p, mix=p.get('mix', 1.0)), np.random.default_rng(seed))
    cases = [('comb', {}), ('comb', {'feedback': -0.95, 'k': 2}), ('comb', {'mode': 'bank'}), ('formant', {}),
             ('formant', {'morph': 'eighth'}), ('sieve', {}), ('fshift', {}), ('fshift', {'k': -1, 'feedback': 0.5}),
             ('freeze', {}), ('freeze', {'every': 'eighth'})]
    for f in (308.87, 462.78, 58.11):
        note(f); x = tone(f)
        for kind, p in cases:
            y = run(kind, x, p)
            check(f'{kind}{p} shape/finite @ {f}', y.shape == x.shape and np.isfinite(y).all())
            c = peaks_cents(y[0], f); check(f'{kind}{p} harmonic peaks @ {f}', c < 1, f'{c:.3f} c')
        # comb resonance itself (impulse response) lands on the note's harmonics
        imp = np.zeros((2, int(2.5 * SR))); imp[:, 100] = 1
        for p in ({'feedback': 0.99}, {'feedback': -0.99}, {'feedback': 0.99, 'k': 2, 'damping': 1.0}, {'feedback': 0.99, 'k': 0.5}):
            c = peaks_cents(run('comb', imp, p)[0], f * min(1, p.get('k', 1)), 0.02, 1.02)
            check(f'comb resonance {p} @ {f}', c < 1, f'{c:.3f} c')
        # sieve pass band centre: multisine on a 0.25 Hz grid (periodic in 4 s) around harmonics 1..3, exact per-line gains
        grid = np.concatenate([np.arange(np.round((h * f - 20) * 4), np.round((h * f + 20) * 4)) / 4 for h in (1, 2, 3)])
        t = np.arange(6 * SR) / SR; ph = np.random.default_rng(1).uniform(0, 2 * np.pi, grid.size)
        ms = sum(np.cos(2 * np.pi * g * t + q) for g, q in zip(grid, ph)) * 0.01
        note(f, 6.0); y = run('sieve', np.stack([ms, ms]), {})[0, SR:5 * SR]
        Y = np.abs(np.fft.rfft(y)); gains = Y[np.round(grid * 4).astype(int)]
        worst = 0.0
        for h in (1, 2, 3):
            sel = np.abs(grid - h * f) <= 20; gg, gh = gains[sel], grid[sel]; keep = gg > 0.3 * gg.max()
            cf = np.polyfit(gh[keep] - h * f, np.log(gg[keep]), 2)
            worst = max(worst, abs(1200 * np.log2((h * f - cf[1] / (2 * cf[0])) / (h * f))))
        check(f'sieve pass band centred on harmonics 1-3 @ {f}', worst < 1, f'{worst:.3f} c')
    # aliasing / inharmonic energy at aggressive settings, 462.78 Hz
    f = 462.78; note(f); x = tone(f); xs = tone(f, ramp=0.3)   # slow onset for fshift: its echo spiral repeats the onset in the window
    for kind, p in [('comb', {'feedback': 0.98, 'damping': 0.0}), ('comb', {'feedback': 0.99, 'k': 2, 'damping': 1.0}), ('comb', {'mode': 'bank', 'feedback': -0.97}),
                    ('formant', {'drive': 8, 'q': 12, 'vowels': ['a']}), ('sieve', {'width_cents': 40}),
                    ('fshift', {'k': 3, 'feedback': 0.7}), ('fshift', {'k': -2}), ('freeze', {'diffusion': 8.0})]:
        db = inharm_db(run(kind, xs if kind == 'fshift' else x, p)[0], f)
        check(f'{kind}{p} inharmonic energy @ {f}', db < -60, f'{db:.1f} dB')
    print('SKIP inharmonic check: fshift mode hz (detunes by design), comb glide (Doppler while the delay sweeps), '
          'formant morph (a moving vowel is AM of the partials, -43 to -96 dB measured; drive aliasing is checked on a static vowel)')
    # determinism, dry before the first note, speed
    note(308.87, 1.0, 0.5); rng = np.random.default_rng(3); x = rng.standard_normal((2, int(1.5 * SR))) * 0.2
    for kind, p in cases + [('fshift', {'mode': 'hz', 'feedback': 0.6}), ('comb', {'glide': 'eighth'})]:
        a, b = run(kind, x, p, 7), run(kind, x, p, 7)
        check(f'{kind}{p} deterministic', np.array_equal(a, b))
        check(f'{kind}{p} dry before first note', np.array_equal(a[:, :int(0.5 * SR)], x[:, :int(0.5 * SR)]))
        t0 = time.perf_counter(); run(kind, x, p); dt = time.perf_counter() - t0
        check(f'{kind}{p} speed (1 s note)', dt < 0.5, f'{dt * 1000:.0f} ms')
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    selftest()
