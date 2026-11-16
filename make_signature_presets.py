"""Writes the signature presets 17-24: chains of the dsp/ modules (synth -> mangle -> resample / re-tune -> space).
Edit a preset's JSON directly afterwards; re-running this script overwrites 17-24."""
import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DOT8 = 0.5413                                    # dotted eighth at 83.13 BPM
VERB = {'type': 'Reverb', 'room_size': 0.55, 'damping': 0.6, 'wet_level': 0.14, 'dry_level': 0.9, 'width': 1.0}

P = {
 '17_fm_snarl': dict(
  description="Signature / growl: a 3-operator FM stack with self-feedback, its index hit on every note and swept by a "
              "saw LFO on the eighth, then a vowel morph (o-a-e) per eighth, multiband grind and OTT. A talking metal snarl",
  engine='fmstack',
  voice=dict(ops=[dict(ratio=1, index=[1.0, 4.5, 0.22], feedback=0.9), dict(ratio=2, index=[1.4, 3.6, 0.3]), dict(ratio=1)],
             algo='serial', lfo_div=1, lfo_shape='saw', lfo_depth=0.45, unison=3, detune=10, spread=0.7,
             drive=2.0, cutoff=7000, attack=0.004, decay=0.5, sustain=0.7, release=0.25),
  fx=[dict(type='formant', vowels=['o', 'a', 'e', 'a', 'o'], morph='eighth', q=6, drive=2.0, mix=0.7),
      dict(type='mbdist', mid_drive_db=10, high_drive_db=6, mix=0.5),
      dict(type='ott', depth=0.35),
      dict(type='HighpassFilter', cutoff_frequency_hz=120), VERB]),

 '18_steel_resample': dict(
  description="Signature / metallic: the bowed-steel preset rendered, cut into a 128-frame wavetable and replayed as a "
              "synth: brightness opens per note, jumps on the eighths, then a comb rings the octave and a metal plate IR",
  engine='resampled',
  voice=dict(source='05_bowed_steel', ref_midi=58, frames=128, order='bright', position=0.3, pos_env=0.5, pos_decay=0.4,
             pos_lfo=0.15, lfo_eighths=4, chop=0.25, unison=5, detune_cents=10, spread=0.8, drive=3,
             cutoff=5000, attack=0.006, decay=0.6, sustain=0.75, release=0.6),
  fx=[dict(type='comb', k=2, feedback=0.85, damping=0.4, mix=0.35),
      dict(type='convolve', kind='metal', decay_s=3, mix=0.3),
      dict(type='HighpassFilter', cutoff_frequency_hz=140)]),

 '19_glass_from_grit': dict(
  description="Signature / glassy: a bit-stepped FM wavetable crushed through stacked clippers, then an STFT sieve keeps "
              "only the note's harmonics, so the grit becomes ringing glass; a spectral freeze holds each note under the next",
  engine='warp',
  voice=dict(table='saw', warps=[dict(mode='fm', amount=3, env=4, decay=0.3, lfo=1), dict(mode='quantize', amount=12)],
             unison=5, detune=18, spread=0.8, drive=2, attack=0.005, decay=0.6, sustain=0.7, release=0.5),
  fx=[dict(type='clipstack', passes=3, drive_db=12),
      dict(type='sieve', width_cents=6, tilt=-2, max_harmonic=30, mix=0.85),
      dict(type='freeze', every='note', hold_s=1.5, diffusion=0.5, mix=0.3),
      dict(type='Supermassive', mode='Cassiopeia', mix=30, delay_ms=300, feedback=60, density=85, width=100,
           lowcut=250, highcut=9000, moddepth=40, modrate=0.25)]),

 '20_phase_zap': dict(
  description="Signature / futuristic: Casio-style resonant phase distortion, the resonance diving from 14x to 2x the "
              "note in 0.12 s, a wavefolder bite on each attack, dotted-eighth echoes",
  engine='pd',
  voice=dict(window='trap', r_start=14, r_end=2, glide=0.12, sub_level=0.25, drive=1.5,
             attack=0.003, decay=0.5, sustain=0.7, release=0.3),
  fx=[dict(type='fold', drive_db=8, env_db=10, env_decay=0.08, mix=0.5),
      dict(type='ott', depth=0.3),
      dict(type='pingpong', time_s=DOT8, feedback=0.4, mix=0.25, tone_hz=3500),
      dict(type='HighpassFilter', cutoff_frequency_hz=120), VERB]),

 '21_harmonic_spiral': dict(
  description="Signature / haunting: a windowed-sync wavetable, then a frequency shifter that moves every partial up by "
              "the note itself, fed back on dotted eighths so each echo climbs another harmonic; glass IR",
  engine='warp',
  voice=dict(table='saw', warps=[dict(mode='sync', amount=2.5, env=3, decay=0.35)], sync_window=1, unison=5, detune=12,
             spread=0.8, cutoff=7000, attack=0.006, decay=0.6, sustain=0.7, release=0.5),
  fx=[dict(type='fshift', mode='harmonic', k=1, feedback=0.55, delay=1.5, mix=0.35),
      dict(type='convolve', kind='glass', mix=0.25),
      dict(type='HighpassFilter', cutoff_frequency_hz=150)]),

 '22_vocal_reese': dict(
  description="Signature / distorted: the reese preset resampled into a wavetable, its position jumping at random on the "
              "eighths, driven inside the table, then a u-o-a vowel morph per note, multiband distortion and OTT",
  engine='resampled',
  voice=dict(source='08_reese_menace', ref_midi=46, position=0.4, pos_env=0.3, chop=0.5, unison=7, detune_cents=14,
             spread=0.8, drive=4, cutoff=3500, f_env_oct=2, res=1.0, attack=0.005, decay=0.6, sustain=0.8, release=0.3),
  fx=[dict(type='formant', vowels=['u', 'o', 'a'], morph='note', q=7, drive=3, mix=0.8),
      dict(type='mbdist'),
      dict(type='ott', depth=0.4),
      dict(type='HighpassFilter', cutoff_frequency_hz=100), dict(type='LowpassFilter', cutoff_frequency_hz=9000)]),

 '23_grain_monolith': dict(
  description="Signature / granular: pitch-synchronous grains scanned through the sync-circuit sound, frozen into a held "
              "spectrum per note, inside a dark sinking cathedral IR. A bed",
  engine='psola',
  voice=dict(source='10_sync_circuit', ref_midi=58, grain_periods=4, scan_speed=0.2, start=0.2, spray=8, stereo_ms=37,
             attack=0.05, decay=1.0, sustain=0.85, release=1.2),
  fx=[dict(type='freeze', every='note', hold_s=3, diffusion=0, crossfade=0.3, mix=0.5),
      dict(type='convolve', kind='cathedral', mix=0.4),
      dict(type='HighpassFilter', cutoff_frequency_hz=180)]),

 '24_gate_circuit': dict(
  description="Signature / electronic: a phase-distorted, 8-step quantized square through a lowpass gate that plucks "
              "on every note, a tuned comb bank (1-2-3x the note) and dotted-eighth ping-pong",
  engine='warp',
  voice=dict(table='square', warps=[dict(mode='pd', amount=0.6, env=0.3, decay=0.2), dict(mode='quantize', amount=8)],
             unison=3, detune=6, spread=0.6, attack=0.003, decay=0.5, sustain=0.75, release=0.3),
  fx=[dict(type='lpg', decay_s=0.22, min_hz=150, max_hz=9000, res=2),
      dict(type='comb', mode='bank', ratios=[1, 2, 3], feedback=0.8, mix=0.3),
      dict(type='pingpong', time_s=DOT8, feedback=0.45, mix=0.3, tone_hz=4000),
      dict(type='HighpassFilter', cutoff_frequency_hz=120), VERB]),
}

for name, d in P.items():
    d = {'description': d.pop('description'), 'engine': d.pop('engine'), 'track': 'melody', 'octave': 0, **d}
    json.dump(d, open(f'presets/{name}.json', 'w'), indent=2)
print('wrote', len(P))
