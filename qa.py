"""Measure the renders: loudness, true peak, loop seam, DC, stereo; per-bar loudness arc for arrangements.

    .venv/Scripts/python qa.py
seam < 1 means the loop's wrap-around jump is no bigger than an ordinary sample step (no click).
"""
import glob, os
import numpy as np, soundfile as sf, pyloudnorm as pyln, scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))


def tp(x): return 20 * np.log10(np.abs(ss.resample_poly(x, 4, 1, axis=0)).max() + 1e-12)


print('sounds (loops)')
for f in sorted(glob.glob(os.path.join(HERE, 'renders/sounds/*_loop.wav'))):
    x, sr = sf.read(f)
    seam = np.abs(x[0] - x[-1]).max() / (np.abs(np.diff(x, axis=0)).max() + 1e-12)
    print(f'  {os.path.basename(f)[:-9]:<22} LUFS {pyln.Meter(sr).integrated_loudness(x):6.1f}  TP {tp(x):5.1f}  seam {seam:4.2f}'
          f'  DC {x.mean():+.4f}  L/R corr {np.corrcoef(x[:, 0], x[:, 1])[0, 1]:4.2f}')

print('arrangements (per-bar LUFS shows the build)')
bar = 8 * 30 / 83.13
for f in sorted(glob.glob(os.path.join(HERE, 'renders/arrangements/*.wav'))):
    x, sr = sf.read(f); M = pyln.Meter(sr)
    arc = [M.integrated_loudness(x[int(b * bar * sr):int((b + 1) * bar * sr)]) for b in range(int(len(x) / sr / bar))]
    print(f'  {os.path.basename(f)[:-4]:<18} LUFS {M.integrated_loudness(x):5.1f}  TP {tp(x):5.1f} | ' + ' '.join(f'{v:4.0f}' for v in arc))
