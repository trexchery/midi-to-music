"""Synth engine: MIDI notes + a JSON preset -> stereo audio. Voices are numpy/numba DSP; effects are pedalboard
built-ins, Valhalla Supermassive (free VST3) and a few numpy effects (tape, pingpong, width).
The free Surge XT VST3 can play a preset's notes instead (engine "surge", voice {"patch": "Pads/Legacy"}).

A preset is {"engine": <voice name>, "voice": {...voice params}, "fx": [{"type": ..., ...}], "octave": 0, ...}.
Voices never change pitch, start or length of a note; they only decide how it sounds.
"""
import glob, os, struct, zlib
import numpy as np, scipy.signal as ss, pedalboard as pb
from numba import njit

SR = 48000
# Optional free VST3 (Valhalla Supermassive). Set SUPERMASSIVE_VST3 to its path if it is installed elsewhere;
# without it, presets that use it fall back to pedalboard's Reverb (close, not identical).
_VST_DIRS = [r'C:\Program Files\Common Files\VST3', '/Library/Audio/Plug-Ins/VST3',
             os.path.expanduser('~/Library/Audio/Plug-Ins/VST3'), os.path.expanduser('~/.vst3'), '/usr/lib/vst3']
VST = {'Supermassive': os.environ.get('SUPERMASSIVE_VST3') or next(
    (os.path.join(d, 'ValhallaSupermassive.vst3') for d in _VST_DIRS if os.path.exists(os.path.join(d, 'ValhallaSupermassive.vst3'))), '')}


# Optional free Surge XT synth (GPL-3.0 VST3 instrument) for presets with "engine": "surge". Found via SURGE_XT_VST3 (the
# .vst3 path), else the standard VST3 folders and ~/tools/surge-xt; its factory patches via SURGE_XT_DATA, else the
# SurgeXTData folder of a portable install or the standard data folders. Missing: those presets are skipped.
_SURGE_DIRS = _VST_DIRS + [os.path.expanduser('~/tools/surge-xt')]


def _find_surge():
    env = os.environ.get('SURGE_XT_VST3')
    for p in [env] if env is not None else [os.path.join(d, s, 'Surge XT.vst3') for d in _SURGE_DIRS for s in ('', 'Surge Synth Team')]:
        if p and os.path.exists(p):
            # pedalboard cannot open a bundle folder on Windows/Linux: use the binary inside it (macOS bundles load as is)
            return next(iter(glob.glob(os.path.join(p, 'Contents', 'x86_64-*', 'Surge XT.*'))), p) if os.path.isdir(p) else p
    return ''


SURGE = _find_surge()


def _surge_patch(name):
    """'Pads/Ghost Pad' -> path of that Surge factory patch (.fxp), or a path to your own .fxp; '' if not found."""
    if os.path.isfile(name): return name
    roots = [r for r in [os.environ.get('SURGE_XT_DATA')] if r]
    d = os.path.dirname(SURGE)
    while d != os.path.dirname(d):                 # portable install: SurgeXTData sits above the plugin
        d, roots = os.path.dirname(d), roots + [os.path.join(d, 'SurgeXTData')]
    roots += [r'C:\ProgramData\Surge XT', '/Library/Application Support/Surge XT', '/usr/share/Surge XT',
              os.path.expanduser('~/.local/share/Surge XT')]
    return next((f for r in roots for f in [os.path.join(r, 'patches_factory', name + '.fxp')] if os.path.isfile(f)), '')


def usable(preset):
    """False for a Surge preset when Surge XT or its patch is not installed (callers skip those)."""
    return preset['engine'] != 'surge' or bool(SURGE and _surge_patch(preset['voice']['patch']))


def hz(m): return 440.0 * 2 ** ((m - 69) / 12)


# Render context shared with dsp/ modules: 'eighth' (s, set by music.py from the MIDI tempo), and per render
# 'notes' [(start_s, dur_s, tuned float midi, vel)] and 'cents'.
CTX = {'eighth': 30 / 83.13, 'notes': [], 'cents': 0.0}


# ---------------------------------------------------------------- building blocks

def adsr(n_on, n_tot, a, d, s, r, sr=SR):
    """Amplitude envelope: attack/decay to sustain while held, exponential-ish release after note-off."""
    t = np.arange(n_tot) / sr
    on = np.where(t < a, (t / max(a, 1e-4)) ** 0.7, s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    t_off = n_on / sr
    lvl = on[min(n_on, n_tot - 1)]
    rel = lvl * np.exp(-(t - t_off) * 6.9 / max(r, 1e-3))     # -60 dB over r seconds
    env = np.where(t < t_off, on, rel)
    fade = min(256, n_tot)
    env[-fade:] *= np.linspace(1, 0, fade)
    return env


def blep(ph, dt):
    out = np.zeros_like(ph)
    m = ph < dt; x = ph[m] / dt[m]; out[m] = 2 * x - x * x - 1
    m = ph > 1 - dt; x = (ph[m] - 1) / dt[m]; out[m] = x * x + 2 * x + 1
    return out


def saw(freq, rng, sr=SR):
    """Band-limited (PolyBLEP) saw for a per-sample frequency array."""
    dt = freq / sr
    ph = (np.cumsum(dt) + rng.random()) % 1.0
    return 2 * ph - 1 - blep(ph, dt)


def square(freq, rng, pw=0.5, sr=SR):
    dt = freq / sr
    ph = (np.cumsum(dt) + rng.random()) % 1.0
    ph2 = (ph + pw) % 1.0
    return (2 * ph - 1 - blep(ph, dt)) - (2 * ph2 - 1 - blep(ph2, dt))


def drift(n, rng, rate=0.5, sr=SR):
    """Smooth random -1..1 wobble (analog drift, vibrato humanizing)."""
    k = max(4, int(n / sr * rate) + 4)
    pts = rng.uniform(-1, 1, k)
    return np.interp(np.linspace(0, k - 3, n), np.arange(k), pts)


def vibrato(n, rng, depth_c, rate, delay, sr=SR):
    t = np.arange(n) / sr
    ramp = np.clip((t - delay) / 0.4, 0, 1)
    return 2 ** ((depth_c * ramp * np.sin(2 * np.pi * rate * t + rng.random() * 6.28) + 3 * drift(n, rng)) / 1200)


@njit(cache=True)
def svf(x, fc, q, mode, sr):
    """Topology-preserving state-variable filter with per-sample cutoff (Simper). mode 0=lp 1=bp 2=hp."""
    y = np.empty_like(x); ic1 = 0.0; ic2 = 0.0; k = 1.0 / q
    for i in range(x.shape[0]):
        f = min(fc[i], 0.45 * sr)
        g = np.tan(np.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k)); a2 = g * a1; a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2
        y[i] = v2 if mode == 0 else (v1 if mode == 1 else x[i] - k * v1 - v2)
    return y


def ladder(x, fc, q=0.9, poles=4, drive=1.0):
    """24 dB/oct lowpass = two cascaded SVFs, soft-driven input for analog warmth."""
    x = np.tanh(drive * x) / np.tanh(drive) if drive > 1 else x
    y = svf(x, fc, q, 0, SR)
    return svf(y, fc, 0.707, 0, SR) if poles == 4 else y


@njit(cache=True)
def karplus(exc, n, period, damp, bright):
    """Karplus-Strong string, fractional delay by linear interpolation; damp=loop gain, bright=0..1."""
    buf = np.zeros(n)
    m = exc.shape[0]
    d = period - 0.5 * (1 - bright)      # the averaging part of the loop filter adds half a sample of delay
    for i in range(n):
        s = exc[i] if i < m else 0.0
        j = i - d
        if j >= 1:
            i0 = int(j); fr = j - i0
            a = buf[i0] * (1 - fr) + buf[i0 + 1] * fr if i0 + 1 < i else buf[i0]
            b = buf[i0 - 1] * (1 - fr) + buf[i0] * fr
            s += damp * (bright * a + (1 - bright) * (0.5 * a + 0.5 * b))
        buf[i] = s
    return buf


def pan(m, p):
    """Equal-power pan, p in -1..1."""
    a = (p + 1) * np.pi / 4
    return np.stack([m * np.cos(a), m * np.sin(a)])


# ---------------------------------------------------------------- voices
# Each voice: (params, f0 Hz, note seconds, velocity 0..1, rng) -> (2, N) float array

def v_additive(p, f, dur, vel, rng):
    rel = p.get('release', 1.5); n_on = int(dur * SR); n = n_on + int(rel * SR)
    t = np.arange(n) / SR
    vib = vibrato(n, rng, p.get('vib_cents', 4), p.get('vib_rate', 4.5), p.get('vib_delay', 0.25))
    if p.get('pitch_drop_cents'):                    # 808-style: starts sharp, falls onto the note
        vib = vib * 2 ** (p['pitch_drop_cents'] * np.exp(-t / p.get('pitch_drop_s', 0.03)) / 1200)
    out = np.zeros((2, n))
    for ratio, amp, dec in p['partials']:          # dec = partial decay time constant (s); 0 = sustain
        if f * ratio > 18000: continue
        ph0 = rng.random() * 6.28                    # same start phase both sides: mono-safe, stereo from detune only
        for ch, det in ((0, -1), (1, 1)):
            fr = f * ratio * 2 ** (det * p.get('spread_cents', 3) / 1200) * vib
            ph = 2 * np.pi * np.cumsum(fr) / SR + ph0
            e = np.exp(-t / dec) if dec else 1.0
            out[ch] += amp * e * np.sin(ph)
    if p.get('chiff', 0):                            # breathy attack noise (pipe / glass rub)
        nz = ss.sosfilt(ss.butter(2, [f * 0.8, min(f * 6, 20000)], 'bandpass', fs=SR, output='sos'), rng.standard_normal(n))
        out += p['chiff'] * nz * np.exp(-t / 0.05)
    return out * adsr(n_on, n, p.get('attack', 0.02), p.get('decay', 0.5), p.get('sustain', 0.8), rel)


def v_fm(p, f, dur, vel, rng):
    """Stacked 2-op FM pairs: [carrier ratio, mod ratio, index, index decay s, amp, amp decay s]."""
    rel = p.get('release', 2.0); n_on = int(dur * SR); n = n_on + int(rel * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for cr, mr, idx, idec, amp, adec in p['pairs']:
        ph0 = rng.random()
        for ch, det in ((0, -1), (1, 1)):
            df = 2 ** (det * p.get('spread_cents', 2) / 1200)
            I = idx * (0.4 + 0.6 * vel) * np.exp(-t / idec)
            mod = I * np.sin(2 * np.pi * f * mr * df * t + ph0)
            out[ch] += amp * np.exp(-t / adec) * np.sin(2 * np.pi * f * cr * df * t + mod)
    return out * adsr(n_on, n, p.get('attack', 0.002), 9, 1.0, rel)


def v_modal(p, f, dur, vel, rng):
    """Inharmonic resonator bank: 'strike' = decaying modes + click, 'bow' = noise-driven resonators."""
    rel = p.get('release', 3.0); n_on = int(dur * SR); n = n_on + int(rel * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    bow = p.get('excite', 'strike') == 'bow'
    if bow:
        src = rng.standard_normal((2, n)) * adsr(n_on, n, p.get('attack', 0.25), 1, 1, p.get('bow_release', 0.3))
    for ratio, amp, dec in p['modes']:
        fm = f * ratio
        if fm > 18000: continue
        ph0 = rng.random() * 6.28
        for ch in (0, 1):
            fr = fm * (1 + rng.uniform(-1, 1) * p.get('mistune', 0.0004))
            if bow:
                b, a = ss.iirpeak(fr, fr / p.get('bw', 2.0), fs=SR)
                out[ch] += amp * ss.lfilter(b, a, src[ch])   # constant Hz bandwidth -> equal energy per mode
            else:
                out[ch] += amp * np.exp(-t / (dec * (0.7 + 0.3 * vel))) * np.sin(2 * np.pi * fr * t + ph0)
    if not bow and p.get('click', 0):
        nz = ss.sosfilt(ss.butter(2, min(f * 8, 16000), 'lowpass', fs=SR, output='sos'), rng.standard_normal(n))
        out += p['click'] * vel * nz * np.exp(-t / 0.004)
    if bow:
        return out
    return out * adsr(n_on, n, p.get('attack', 0.001), 9, 1.0, rel)


def v_subtractive(p, f, dur, vel, rng):
    """Detuned PolyBLEP saws/squares + sub -> driven 4-pole lowpass with its own envelope."""
    rel = p.get('release', 0.6); n_on = int(dur * SR); n = n_on + int(rel * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    k = p.get('voices', 5); det = p.get('detune_cents', 12)
    for v in range(k):
        off = det * (v / (k - 1) * 2 - 1) if k > 1 else 0
        fr = f * 2 ** ((off + 4 * drift(n, rng, 0.3)) / 1200) * np.ones(n)
        osc = saw(fr, rng) if p.get('wave', 'saw') == 'saw' else square(fr, rng, p.get('pw', 0.5))
        out += pan(osc, (v / (k - 1) * 2 - 1) * p.get('width', 0.8) if k > 1 else 0) / np.sqrt(k)
    if p.get('sub', 0):
        out += p['sub'] * np.sin(2 * np.pi * f / 2 * t)
    fe = adsr(n_on, n, p.get('f_attack', 0.005), p.get('f_decay', 0.4), p.get('f_sustain', 0.3), rel)
    fc = p.get('cutoff', 400) * 2 ** (p.get('f_env_oct', 4) * fe * (0.5 + 0.5 * vel))
    for ch in (0, 1):
        out[ch] = ladder(out[ch], fc, p.get('res', 0.9), 4, p.get('drive', 1.5))
    return out * adsr(n_on, n, p.get('attack', 0.01), p.get('decay', 0.6), p.get('sustain', 0.7), rel)


def v_formant(p, f, dur, vel, rng):
    """Breathy detuned voices through a vowel formant bank (ghost choir)."""
    rel = p.get('release', 1.2); n_on = int(dur * SR); n = n_on + int(rel * SR)
    out = np.zeros((2, n))
    k = p.get('voices', 6)
    for v in range(k):
        vib = vibrato(n, rng, p.get('vib_cents', 14), 4.6 + rng.uniform(-0.6, 0.6), 0.2)
        fr = f * 2 ** (rng.uniform(-1, 1) * p.get('detune_cents', 9) / 1200) * vib
        src = saw(fr, rng) + p.get('breath', 0.25) * rng.standard_normal(n)
        sig = np.zeros(n)
        for fq, bw, g in p['formants']:
            b, a = ss.iirpeak(fq, fq / bw, fs=SR)
            sig += g * ss.lfilter(b, a, src)
        out += pan(sig, rng.uniform(-0.9, 0.9)) / np.sqrt(k)
    return out * adsr(n_on, n, p.get('attack', 0.18), 1, 1.0, rel)


def v_granular(p, f, dur, vel, rng):
    """Grain cloud cut from a soft harmonic source tone; grains jitter in time, pan and fine pitch."""
    rel = p.get('release', 2.0); n_on = int(dur * SR); n = n_on + int(rel * SR)
    src_len = int(0.5 * SR)
    ts = np.arange(src_len) / SR
    src = sum(a * np.sin(2 * np.pi * f * h * ts) for h, a in p.get('source', [[1, 1], [2, .4], [3, .25], [4, .12], [5, .08]]) if f * h < 18000)
    env = adsr(n_on, n, p.get('attack', 0.08), 1, 1.0, rel)
    out = np.zeros((2, n + SR))
    for _ in range(int(p.get('density', 90) * n / SR)):
        st = int(rng.uniform(0, n) * 1.0)
        if env[min(st, n - 1)] < 0.01: continue
        gl = int(rng.uniform(*p.get('grain_ms', [30, 140])) / 1000 * SR)
        r = rng.choice(p.get('ratios', [1, 1, 1, 2, 0.5])) * 2 ** (rng.normal(0, p.get('jitter_cents', 6)) / 1200)
        pos = rng.uniform(0, src_len - gl * r - 2)
        g = np.interp(pos + np.arange(gl) * r, np.arange(src_len), src) * np.hanning(gl)
        out[:, st:st + gl] += pan(g, rng.uniform(-1, 1)) * env[min(st, n - 1)]
    return out[:, :n] * 6 / np.sqrt(p.get('density', 90))


def v_sync(p, f, dur, vel, rng):
    """Hard-sync saw (4x oversampled) with a sweeping slave ratio: the 'futuristic' tear."""
    rel = p.get('release', 0.5); n_on = int(dur * SR); n = n_on + int(rel * SR)
    os_ = 4; N = n * os_
    t = np.arange(N) / (SR * os_)
    ratio = p.get('ratio_end', 1.5) + (p.get('ratio_start', 4.0) - p.get('ratio_end', 1.5)) * np.exp(-t / p.get('sweep', 0.25))
    out = np.zeros((2, n))
    for ch, det in ((0, -1), (1, 1)):
        fr = f * 2 ** (det * p.get('spread_cents', 6) / 1200)
        pm = (np.cumsum(np.full(N, fr / (SR * os_))) + rng.random()) % 1.0
        slave = 2 * ((pm * ratio) % 1.0) - 1
        out[ch] = ss.resample_poly(slave, 1, os_)[:n]
    fe = adsr(n_on, n, 0.003, p.get('f_decay', 0.3), 0.4, rel)
    for ch in (0, 1):
        out[ch] = ladder(out[ch], p.get('cutoff', 1200) * 2 ** (3 * fe), p.get('res', 1.2), 2)
    return out * adsr(n_on, n, p.get('attack', 0.004), 0.5, p.get('sustain', 0.7), rel)


def v_pluck(p, f, dur, vel, rng):
    """Karplus-Strong string with a body filter; ring time set by 'ring' seconds."""
    rel = p.get('release', 1.0); n_on = int(dur * SR); n = n_on + int(rel * SR)
    out = np.zeros((2, n))
    period = SR / f
    damp = 10 ** (-3 / (p.get('ring', 3.0) * f))           # -60 dB after ring seconds
    exc = rng.uniform(-1, 1, int(period) + 1)
    exc = ss.lfilter([1 - p.get('pick_dark', 0.5)], [1, -p.get('pick_dark', 0.5)], exc) * vel
    for ch, det in ((0, -1), (1, 1)):
        y = karplus(exc, n, period * 2 ** (det * p.get('spread_cents', 2) / 1200), damp, p.get('bright', 0.3))
        out[ch] = y
    return out * adsr(n_on, n, 0.001, 9, 1.0, rel)


VOICES = {'additive': v_additive, 'fm': v_fm, 'modal': v_modal, 'subtractive': v_subtractive,
          'formant': v_formant, 'granular': v_granular, 'sync': v_sync, 'pluck': v_pluck}


# ---------------------------------------------------------------- effects

def fx_tape(x, p, rng):
    """Wow/flutter via a modulated delay, gentle saturation, optional hiss: the haunted-cassette sound."""
    n = x.shape[1]; t = np.arange(n) / SR
    d = (p.get('base_ms', 8) + p.get('wow_ms', 2.0) * np.sin(2 * np.pi * p.get('wow_hz', 0.6) * t)
         + p.get('flutter_ms', 0.15) * np.sin(2 * np.pi * 9 * t) + p.get('wow_ms', 2.0) * 0.6 * drift(n, rng, 1.5)) / 1000 * SR
    idx = np.arange(n) - d
    y = np.stack([np.interp(idx, np.arange(n), c) for c in x])
    dr = p.get('drive', 1.5)
    y = np.tanh(dr * y) / np.tanh(dr)
    return y + p.get('hiss', 0.0) * ss.sosfilt(ss.butter(2, 3000, 'highpass', fs=SR, output='sos'), rng.standard_normal(y.shape))


def fx_pingpong(x, p, rng):
    """Stereo ping-pong echoes, each repeat darker than the last."""
    d = int(p.get('time_s', 0.36) * SR); fb = p.get('feedback', 0.45); y = x.copy()
    mono = x.mean(0); sos = ss.butter(1, p.get('tone_hz', 3500), 'lowpass', fs=SR, output='sos')
    echo = mono
    for k in range(1, p.get('repeats', 8) + 1):
        if k * d >= x.shape[1]: break
        echo = ss.sosfilt(sos, echo) * fb
        ch = k % 2
        y[ch, k * d:] += p.get('mix', 0.35) * echo[:x.shape[1] - k * d]
    return y


def fx_width(x, p, rng):
    m, s = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2 * p.get('amount', 1.3)
    return np.stack([m + s, m - s])


NUMPY_FX = {'tape': fx_tape, 'pingpong': fx_pingpong, 'width': fx_width}


_PLUGS = {}


def set_param(plug, k, v):
    """pedalboard's setattr scans every value of a VST parameter (seconds each); binary-search the raw 0..1 value."""
    par = plug.parameters[k]
    if isinstance(v, str) or isinstance(v, bool):
        par.raw_value = par.get_raw_value_for(v)
    else:
        lo, hi = 0.0, 1.0
        for _ in range(30):
            par.raw_value = mid = (lo + hi) / 2
            lo, hi = (mid, hi) if float(getattr(plug, k)) < v else (lo, mid)
        par.raw_value = hi                     # first raw value whose display reaches v
    got = getattr(plug, k)
    if (got != v) if isinstance(v, (str, bool)) else abs(float(got) - v) > 0.01 * max(1, abs(v)):
        raise ValueError(f'{k}: wanted {v}, plugin reports {got}')


def apply_fx(x, chain, rng):
    for spec in chain:
        spec = dict(spec); kind = spec.pop('type')
        if kind in NUMPY_FX:
            x = NUMPY_FX[kind](x, spec, rng)
        elif kind in VST and not os.path.exists(VST[kind]):
            if kind not in _PLUGS:
                print(f'note: {kind} VST3 not found, using pedalboard Reverb instead (see README)'); _PLUGS[kind] = None
            mix = spec.get('mix', 40) / 100
            x = pb.Reverb(room_size=0.92, damping=0.5, wet_level=0.7 * mix, dry_level=1 - 0.5 * mix, width=1.0)(
                x.astype(np.float32), SR).astype(np.float64)
        elif kind in VST:
            if kind not in _PLUGS:             # loading + parameter scan costs ~8 s, so load once per process
                _PLUGS[kind] = pb.load_plugin(VST[kind])
            plug = _PLUGS[kind]
            plug.reset()                       # clear the previous render's tail
            for k, v in spec.items():
                set_param(plug, k, v)
            x = plug(x.astype(np.float32), SR).astype(np.float64)
        else:
            x = getattr(pb, kind)(**spec)(x.astype(np.float32), SR).astype(np.float64)
    return x


# ---------------------------------------------------------------- render

_JUCE_B64 = '.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+'   # JUCE's own base64 alphabet, LSB first
_SURGE_STATE = {}


def _juce_b64(b):
    out, bits, nb = [], 0, 0
    for x in b:
        bits |= x << nb; nb += 8
        while nb >= 6: out.append(_JUCE_B64[bits & 63]); bits >>= 6; nb -= 6
    return f'{len(b)}.' + ''.join(out + ([_JUCE_B64[bits]] if nb else []))


def _surge_load(patch):
    """Load the Surge plugin once per process (~30 s: pedalboard scans its 775 parameters) and put `patch` (.fxp) into it
    (about 1 s per later patch change). pedalboard can only load .vstpreset, so the
    patch data (an .fxp's chunk is Surge's own patch format) is written into the plugin's VST3 state, which wraps it in
    JUCE base64 inside an XML."""
    if 'plug' not in _SURGE_STATE: _SURGE_STATE['plug'] = pb.load_plugin(SURGE)
    plug = _SURGE_STATE['plug']
    if _SURGE_STATE.get('patch') != patch:
        with open(patch, 'rb') as f: d = f.read()
        raw = plug.raw_state                           # 'VC2!' + length + <VST3PluginState><IComponent>patch</IComponent>...\0
        xml = raw[8:-1].decode('latin1'); i, j = xml.index('<IComponent>') + 12, xml.index('</IComponent>')
        xml = (xml[:i] + _juce_b64(d[60:60 + struct.unpack('>I', d[56:60])[0]]) + xml[j:]).encode('latin1')
        plug.raw_state = b'VC2!' + struct.pack('<I', len(xml) + 1) + xml + b'\0'
        plug([], 0.1, SR)                              # the patch is only applied by the first process call
        for k in ('a_pitch_bend_up_range', 'a_pitch_bend_down_range', 'b_pitch_bend_up_range', 'b_pitch_bend_down_range'):
            set_param(plug, k, '2 semitones')          # known range, so cents can be done with pitch bend
        _SURGE_STATE['patch'] = patch
    return plug


def _surge_dry(notes, preset, n, cents):
    """The whole note list at once as MIDI through Surge XT (pedalboard's instrument API) -> (2, n). Velocity is passed
    on as is (preset 'vel_curve' does not apply); `cents` is a pitch bend of +-2 semitones, 14 bit (0.025 cent steps).
    Surge's own randomness (unison/oscillator phase, noise) makes two renders differ slightly: not sample-identical."""
    plug = _surge_load(_surge_patch(preset['voice']['patch']))
    plug([(bytes([0xB0, 120, 0]), 0)], 0.25, SR)   # all sound off + silent tail: pedalboard's reset leaves held/ringing voices that bleed into this render
    bend = int(np.clip(round(8192 + cents / 200 * 8192), 0, 16383))
    ev = [(0, 0, bytes([0xE0, bend & 127, bend >> 7]))]
    for st, du, m, v in notes:
        m = int(round(m + 12 * preset.get('octave', 0)))
        if st < n / SR and 0 <= m < 128:           # out-of-range notes (octave shift) are dropped
            ev += [(st, 1, bytes([0x90, m, int(np.clip(round(v), 1, 127))])), (min(max(st + du, st + 0.005), n / SR), 0, bytes([0x80, m, 0]))]   # >= 5 ms: off sorts before on at equal times
    # off before on at equal times; bend first
    y = plug([(b, t) for t, _, b in sorted(ev, key=lambda e: e[:2])], n / SR, SR).astype(np.float64)
    return np.pad(y, ((0, 0), (0, max(0, n - y.shape[1]))))[:, :n]


def render(notes, preset, length_s, seed=0, period=None, cents=0.0):
    """notes: [(start_s, dur_s, midi, vel 0-127)] -> (2, length) audio, dry voices then the preset's FX chain.
    period: reuse per-note randomness every `period` notes, so every pass of a loop is sample-identical.
    cents: tuning offset from A440 (the source melody sits at -12.6 c, its bass at about -4.5 c)."""
    n = int(length_s * SR)
    # context for note-aware voices/effects (dsp/): tuned float MIDI pitch per note, as it will sound
    CTX.update(cents=cents, notes=[(st, du, m + 12 * preset.get('octave', 0) + cents / 100, v) for st, du, m, v in notes])
    surge = preset['engine'] == 'surge'        # Surge XT plays the whole note list itself; other engines render per note
    voice = None if surge else VOICES[preset['engine']]
    dry = _surge_dry(notes, preset, n, cents) if surge else np.zeros((2, n))
    base = zlib.crc32(preset['name'].encode()) + seed
    for i, (st, du, m, v) in enumerate([] if surge else notes):
        rng = np.random.default_rng(base + (i % period if period else i))
        a = voice(preset.get('voice', {}), hz(m + 12 * preset.get('octave', 0) + cents / 100), du, v / 127, rng)
        a *= (v / 127) ** preset.get('vel_curve', 1.4)
        i0 = int(round(st * SR))
        if i0 >= n: continue
        k = min(a.shape[1], n - i0)
        dry[:, i0:i0 + k] += a[:, :k]
    dry = ss.sosfilt(ss.butter(2, 20, 'highpass', fs=SR, output='sos'), dry)   # DC blocker (FM 1:1 and sync make DC)
    pk = np.abs(dry).max()
    if pk > 0:
        dry *= 0.5 / pk                       # fixed headroom into the FX so drive amounts mean the same everywhere
    return apply_fx(dry, preset.get('fx', []), np.random.default_rng(base + 999))


@njit(cache=True)
def _release(g, coef):
    out = np.empty_like(g); cur = 1.0
    for i in range(g.shape[0]):
        cur = min(g[i], cur + (1.0 - cur) * coef)      # instant attack (target is pre-smoothed), slow recovery
        out[i] = cur
    return out


def limit(x, ceiling_db=-1.0, look_ms=5.0, release_ms=150.0):
    """Transparent offline look-ahead peak limiter (no makeup gain). Checks peaks on 4x oversampled audio."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    c = 10 ** (ceiling_db / 20)
    pk = np.abs(ss.resample_poly(x, 4, 1, axis=1)).max(0).reshape(-1, 4).max(1)[:x.shape[1]]
    tgt = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = int(look_ms / 1000 * SR)
    g = uniform_filter1d(minimum_filter1d(tgt, 2 * L + 1), L)
    g = _release(g, 1 - np.exp(-1 / (release_ms / 1000 * SR)))
    return np.clip(x * g, -c, c)


# ---------------------------------------------------------------- advanced modules
# Every module in dsp/ may define VOICES = {name: fn(p, f, dur, vel, rng)} and FX = {name: fn(x, p, rng)};
# they are registered here, so a new module needs no edit to this file.
import importlib, pkgutil, dsp  # noqa: E402  (after everything dsp modules use from this file)
for _m in pkgutil.iter_modules(dsp.__path__):
    if _m.name.startswith('_'): continue
    _mod = importlib.import_module(f'dsp.{_m.name}')
    VOICES.update(getattr(_mod, 'VOICES', {})); NUMPY_FX.update(getattr(_mod, 'FX', {}))
