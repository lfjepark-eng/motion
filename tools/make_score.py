"""30 s score for "종이 한 장": 128 BPM, D minor, 16 bars, cues locked to SPEC.md.

usage: make_score.py <out.wav> <sfx_dir> <ffmpeg_exe>
Everything is seeded; the bundled Pixabay riser/impacts are decoded with ffmpeg.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 48000
DUR = 30.0
N = int(SR * DUR)
BEAT = 60 / 128
BAR = BEAT * 4
FRAME = 1 / 30
rng = np.random.default_rng(1031)

OUT, SFX_DIR, FFMPEG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

# Cue times, mirrored from SPEC.md / the compositions.
TEAR = [20 * BEAT, 32 * BEAT, 44 * BEAT]  # 9.375, 15.0, 20.625
FINAL = 56 * BEAT  # 26.25
DRUMS_IN, DROP = TEAR[1], TEAR[2]


def b(beats):
    return beats * BEAT


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def buf():
    return np.zeros((N, 2), np.float32)


def place(dst, mono_or_st, t, gain=1.0, pan=0.0):
    """Add a clip at time t (s). pan -1..1 with equal-power law."""
    x = np.asarray(mono_or_st, np.float32)
    if x.ndim == 1:
        lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        x = np.stack([x * lg * 1.414, x * rg * 1.414], axis=1)
    i = int(round(t * SR))
    if i < 0:
        x, i = x[-i:], 0
    j = min(N, i + len(x))
    if j > i:
        dst[i:j] += x[: j - i] * gain


def env_exp(n, attack, decay):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


def bandpass(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], "bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def lowpass(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "lowpass", fs=SR, output="sos"), x, axis=0)


def highpass(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "highpass", fs=SR, output="sos"), x, axis=0)


def noise(n):
    return rng.standard_normal(n).astype(np.float32)


def reverb(x, seconds=2.4, mix=0.35, seed=7):
    """Stereo convolution reverb from decaying noise (different seed per side)."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    irs = []
    for _ in range(2):
        ir = r.standard_normal(n) * np.exp(-t * 6.9 / seconds)
        ir = lowpass(ir, 6500)
        ir[: int(0.012 * SR)] = 0  # pre-delay
        irs.append(ir / np.sqrt(np.sum(ir**2)))
    mono = x.mean(axis=1) if x.ndim == 2 else x
    wet = np.stack([signal.fftconvolve(mono, irs[k])[:N] for k in range(2)], axis=1)
    dry = x if x.ndim == 2 else np.stack([x, x], axis=1)
    return dry * (1 - mix) + wet * mix * 1.6


# ---------------------------------------------------------------- instruments
def piano(m, dur=3.0, vel=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(m)
    out = np.zeros(n, np.float32)
    for k, amp in enumerate([1.0, 0.42, 0.2, 0.09, 0.05], start=1):
        fk = f * k * (1 + 0.0004 * k * k)  # slight inharmonicity
        if fk > 12000:
            break
        out += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t * (0.9 + 0.55 * k))
    out *= np.clip(t / 0.006, 0, 1)  # felt hammer: soft 6 ms attack
    hammer = lowpass(noise(int(0.02 * SR)), 1800) * env_exp(int(0.02 * SR), 0.001, 0.004) * 0.25
    out[: len(hammer)] += hammer
    return lowpass(out, 2600 + 2400 * vel) * vel


def saw(f, n, phase=0.0):
    t = np.arange(n) / SR
    return (2 * ((f * t + phase) % 1.0) - 1).astype(np.float32)


def supersaw(m, dur, voices=7, detune=0.18, seed=0):
    n = int(dur * SR)
    r = np.random.default_rng(seed)
    L, R = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for v in range(voices):
        cents = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune * 100
        f = mtof(m) * 2 ** (cents / 1200)
        s = saw(f, n, r.random())
        pan = (v / (voices - 1)) * 2 - 1
        L += s * np.cos((pan + 1) * np.pi / 4)
        R += s * np.sin((pan + 1) * np.pi / 4)
    return np.stack([L, R], axis=1) / voices


def kick():
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.13)
    click = highpass(noise(n), 2500) * np.exp(-t / 0.004) * 0.35
    return np.tanh(1.6 * (body + click)) * 0.95


def clap():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    x = bandpass(noise(n), 900, 3200)
    e = np.zeros(n)
    for k, d in enumerate([0.0, 0.011, 0.022]):
        e += np.where(t >= d, np.exp(-(t - d) / 0.006), 0) * (0.7 if k < 2 else 1.0)
    e += np.where(t >= 0.022, np.exp(-(t - 0.022) / 0.09), 0) * 0.45
    return x * e * 0.9


def hat(open_=False):
    n = int((0.32 if open_ else 0.05) * SR)
    t = np.arange(n) / SR
    x = highpass(noise(n), 7500, order=4)
    return x * np.exp(-t / (0.11 if open_ else 0.013)) * 0.5


def crash(dur=2.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = highpass(noise(n), 4200, order=3) + 0.3 * bandpass(noise(n), 1500, 4000)
    return x * np.exp(-t / 0.65) * np.clip(t / 0.002, 0, 1) * 0.45


def paper_tear(dur=0.85, peak_at=0.40, seed=11):
    """Fibrous rip: a sparse crackle that thickens into the give, then a torn swoosh."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    rate = 60 + 900 * np.exp(-((t - peak_at) / 0.09) ** 2) + 220 * np.clip((t - peak_at) / 0.05, 0, 1) * np.exp(-np.clip(t - peak_at, 0, None) / 0.18)
    imp = (r.random(n) < rate / SR).astype(np.float32) * r.uniform(0.3, 1.0, n)
    grain = signal.lfilter([1.0], [1.0, -0.86], imp)  # each tick rings ~5 ms
    crackle = bandpass(grain * r.standard_normal(n), 900, 7500)
    swoosh = bandpass(r.standard_normal(n), 350, 4200) * np.clip((t - peak_at + 0.015) / 0.015, 0, 1) * np.exp(-np.clip(t - peak_at, 0, None) / 0.16)
    body = lowpass(r.standard_normal(n), 220) * np.exp(-np.clip(t - peak_at, 0, None) / 0.07) * (t >= peak_at)
    x = 1.0 * crackle / (np.abs(crackle).max() + 1e-6) + 0.55 * swoosh / (np.abs(swoosh).max() + 1e-6) + 0.5 * body / (np.abs(body).max() + 1e-6)
    return x * np.clip(t / 0.03, 0, 1) * 0.8


def scratch(dur, seed, lo=1800, hi=7000):
    """Brush/pen friction: band-limited noise with jittery pressure."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    pressure = uniform_filter1d(r.random(n), int(0.004 * SR)) ** 2
    x = bandpass(r.standard_normal(n), lo, hi) * pressure
    shape = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.6
    return x * shape / (np.abs(x).max() + 1e-6)


def load_sfx(name):
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "x.wav"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(SFX_DIR / name), "-ar", str(SR), "-ac", "2", str(wav)], check=True)
        _, x = wavfile.read(wav)
    x = x.astype(np.float32) / 32768.0
    return x / (np.abs(x).max() + 1e-6)


def place_peak(dst, clip, t_picture, gain):
    """Land a clip's loudest point one frame after its picture (SPEC: sound lags light)."""
    env = uniform_filter1d(np.abs(clip).mean(axis=1), int(0.01 * SR))
    pk = int(np.argmax(env)) / SR
    place(dst, clip, t_picture + FRAME - pk, gain)


def place_hit(dst, clip, t_picture, gain, keep=1.2):
    """Impacts: the bundled ones sustain ~2 s at full level, so land the ONSET on the
    picture and cut them to `keep` seconds with an exponential tail."""
    env = uniform_filter1d(np.abs(clip).mean(axis=1), int(0.005 * SR))
    on = int(np.argmax(env > 0.3 * env.max()))
    n = int(keep * SR)
    hit = clip[on : on + n] * np.exp(-np.arange(min(n, len(clip) - on)) / SR / (keep / 3))[:, None]
    place(dst, hit, t_picture + FRAME, gain)


# ---------------------------------------------------------------- arrangement
ambient, keys, perc, bass, synth, fx = (buf() for _ in range(6))

# Wind: slow-breathing band noise, full in the quiet, receding under the bass.
t_all = np.arange(N) / SR
wind = bandpass(noise(N), 180, 1400) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.11 * t_all + 1.0))
wind_env = np.interp(t_all, [0, 1.2, TEAR[0], TEAR[0] + 0.3, TEAR[1], DUR], [0.55, 1.0, 1.0, 0.35, 0.0, 0.0])
ambient += np.stack([wind * wind_env, np.roll(wind, 2400) * wind_env], axis=1) * 0.08

# Drone pad D2+A2, under the opening only.
pad = sum(np.sin(2 * np.pi * mtof(m) * t_all) for m in (38, 45, 50)) / 3
pad_env = np.interp(t_all, [0, 3, TEAR[0] - 0.4, TEAR[0], TEAR[1], DUR], [0, 1, 1, 0.5, 0.0, 0.0])
ambient += np.stack([pad, pad], axis=1) * pad_env[:, None] * 0.05

# Felt piano motif (beats, midi, velocity): quiet opening, echoed in the coda.
motif = [(0, 62, .45), (1.5, 69, .30), (3, 65, .36), (4, 74, .28), (6, 72, .32), (7, 69, .26),
         (8, 62, .40), (9.5, 69, .28), (11, 65, .34), (12, 77, .26), (14, 76, .28), (15, 74, .24),
         (16, 62, .34), (17, 65, .26), (18, 69, .26)]
for bt, m, v in motif:
    place(keys, piano(m, 3.2, v), b(bt), 1.0, pan=(m - 66) / 24)
# Through the kraft layer the motif keeps going, thinner, over the bass.
for bt, m, v in [(21, 74, .22), (23, 69, .2), (25, 77, .22), (27, 76, .2), (29, 74, .2), (30, 69, .18)]:
    place(keys, piano(m, 2.6, v), b(bt), 1.0, pan=0.2)
# Coda: the opening's first three notes, after the last hit.
for bt, m, v in [(58, 62, .3), (59.5, 69, .22), (61, 65, .26)]:
    place(keys, piano(m, 3.0, v), b(bt), 1.0)

# Reverse swell into tear 1: a D minor chord played backwards.
chord = sum(piano(m, 1.6, 0.5) for m in (50, 57, 62, 65, 69))
swell = chord[::-1] * np.linspace(0, 1, len(chord)) ** 2
place(keys, swell, TEAR[0] - len(swell) / SR + FRAME, 0.55)

keys = reverb(keys, 2.8, 0.42, seed=3)
ambient = reverb(ambient, 1.8, 0.25, seed=5)

# Brush strokes on hanji: one stroke per character (timings from s1-hanji.html).
brush_chars = [0.0 + i * 0.18 for i in range(3)] + [b(4) + i * 0.18 for i in range(8)] + [b(9) + i * 0.18 for i in range(2)]
for k, t0 in enumerate(brush_chars):
    place(fx, scratch(0.34, 100 + k, 1200, 5200), max(0.0, t0), 0.10, pan=-0.3 + 0.06 * (k % 10))
# Pen on kraft: "그", "고요를", underline, memo.
pen_chars = [b(21)] + [b(22) + i * 0.12 for i in range(3)]
for k, t0 in enumerate(pen_chars):
    place(fx, scratch(0.18, 200 + k, 2500, 8000), t0, 0.12, pan=0.15)
place(fx, scratch(0.55, 250, 2200, 7000), b(25), 0.12, pan=0.1)  # underline
place(fx, scratch(0.22, 251, 2600, 8000), b(27), 0.10, pan=0.35)  # 쉿

# Paper tears at all three holes, peak one frame after the picture gives.
for k, T in enumerate(TEAR):
    rip = paper_tear(seed=11 + k)
    place(fx, rip, T - 0.40 + FRAME, 0.55 + 0.1 * k, pan=0.0)

# Bundled Pixabay hits: the riser once into the payoff, impacts on the payoff and the last line.
place_peak(fx, load_sfx("riser.mp3"), DROP, 0.6)
place_hit(fx, load_sfx("impact-bass-1.mp3"), DROP, 0.7)
place_hit(fx, load_sfx("impact-bass-2.mp3"), FINAL, 0.75, keep=2.4)
place_hit(fx, load_sfx("impact-bass-2.mp3"), TEAR[0], 0.35, keep=0.9)

# Bass from tear 1: root per bar Dm | Bb | F | C, 8th pulses, a saw an octave up for phones.
roots = [38, 34, 41, 36]
for bar in range(5, 14):
    if TEAR[0] + 0.0 > bar * BAR + 0.001 and bar < 5:
        continue
    root = roots[(bar - 5) % 4]
    for e in range(8):
        t0 = bar * BAR + e * BEAT / 2
        if t0 < TEAR[0] - 1e-6 or t0 >= FINAL:
            continue
        n = int(BEAT / 2 * SR * 0.92)
        tt = np.arange(n) / SR
        sub = np.sin(2 * np.pi * mtof(root - 12) * tt)
        grit = lowpass(saw(mtof(root), n), 1300 if t0 < DROP else 2600)
        note = (0.5 * sub + 0.75 * grit) * np.clip(tt / 0.004, 0, 1) * np.exp(-tt / 0.22)
        place(bass, np.tanh(1.4 * note), t0, 0.5 if t0 < DRUMS_IN else 0.62)

# Drums from tear 2. Snare/clap roll as the fill into tear 2.
K, C, H, HO, CR = kick(), clap(), hat(), hat(True), crash()
for i in range(32, 56):
    t0 = b(i)
    place(perc, K, t0, 0.62)
    if i % 2 == 1:
        place(perc, C, t0, 0.8, pan=0.05)
    for s in range(4):
        place(perc, H, t0 + s * BEAT / 4, (0.5 if s % 2 else 0.26) * (1.0 if t0 >= DROP else 0.75), pan=0.25)
    if t0 >= DROP:
        place(perc, HO, t0 + BEAT / 2, 0.34, pan=-0.2)
fill_t = [b(30) + k * BEAT / 4 for k in range(4)] + [b(31) + k * BEAT / 8 for k in range(8)]
for k, t0 in enumerate(fill_t):
    place(perc, C, t0, 0.18 + 0.32 * k / len(fill_t), pan=0.05)
place(perc, CR, DROP, 0.6)
place(perc, CR, FINAL, 0.7)
place(perc, K, FINAL, 1.0)

# Drop: supersaw offbeat stabs on the progression, plus a lead hook.
chords = {38: (62, 65, 69, 74), 34: (58, 62, 65, 70), 41: (60, 65, 69, 72), 36: (60, 64, 67, 72)}
for i in range(44, 56):
    root = roots[((i // 4) - 5) % 4]
    for half in (0.5,):
        t0 = b(i + half)
        n = int(BEAT * 0.6 * SR)
        st = sum(supersaw(m, BEAT * 0.6, seed=m) for m in chords[root]) / 2
        st *= np.exp(-np.arange(n) / SR / 0.22)[:, None]
        place(synth, lowpass(st, 6500), t0, 0.62)
lead = [(44, 74), (45, 77), (46, 81), (47, 79), (48, 77), (48.5, 76), (49, 74), (50, 72),
        (52, 74), (53, 77), (54, 81), (54.5, 84), (55, 86)]
for bt, m in lead:
    n = int(BEAT * 0.9 * SR)
    tt = np.arange(n) / SR
    tone = sum(saw(mtof(m) * d, n) for d in (1.0, 1.004, 0.996)) / 3
    tone = lowpass(tone, 3800) * np.clip(tt / 0.005, 0, 1) * np.exp(-tt / 0.3)
    place(synth, tone, b(bt), 0.4, pan=0.1)
# Last line: a sustained Dm wall that decays into the coda.
wall_n = int((DUR - FINAL) * SR)
wall = sum(supersaw(m, DUR - FINAL, seed=m + 1) for m in (50, 62, 65, 69, 74)) / 2.5
wall = lowpass(wall, 3000) * np.exp(-np.arange(wall_n) / SR / 1.1)[:, None]
place(synth, wall, FINAL, 0.55)
synth = reverb(synth, 1.6, 0.22, seed=9)

# Sidechain: everything melodic ducks under each kick.
duck = np.ones(N, np.float32)
for i in range(32, 56):
    k0 = int(b(i) * SR)
    seg = np.arange(int(BEAT * SR)) / SR
    g = 1 - 0.55 * np.exp(-seg / 0.11)
    j = min(N, k0 + len(g))
    duck[k0:j] = np.minimum(duck[k0:j], g[: j - k0])
bass *= duck[:, None]
synth *= duck[:, None]
keys *= (0.4 + 0.6 * duck)[:, None]

mix = ambient + keys * 0.9 + perc + bass + synth + fx
mix = highpass(mix, 32)

# End on the bar line: 0.5 s fade into 30.0.
fade = np.clip((DUR - t_all) / 0.5, 0, 1)
mix *= fade[:, None]


# ---------------------------------------------------------------- loudness
def lufs(x):
    """BS.1770-4 integrated loudness (48 kHz K-weighting, gated)."""
    s1 = signal.lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585], x, axis=0)
    k = signal.lfilter([1, -2, 1], [1, -1.99004745483398, 0.99007225036621], s1, axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = np.array([np.mean(k[i : i + blk] ** 2, axis=0).sum() for i in range(0, len(k) - blk, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[lk > -70]
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g = ms[(lk > -70) & (lk > rel)]
    return -0.691 + 10 * np.log10(g.mean())


def limit(x, ceiling):
    """Lookahead peak limiter: 3 ms lookahead, 60 ms release."""
    peak = maximum_filter1d(np.abs(x).max(axis=1), int(0.003 * SR) * 2 + 1)
    gr = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    gr = np.minimum(gr, uniform_filter1d(gr, int(0.003 * SR)))
    rel = signal.lfilter([1 - np.exp(-1 / (0.06 * SR))], [1, -np.exp(-1 / (0.06 * SR))], gr)
    gr = np.minimum(gr, rel)
    return x * gr[:, None]


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max())


TARGET, CEIL = -14.0, 10 ** (-1.6 / 20)
y = mix / (np.abs(mix).max() + 1e-9) * 0.5
for _ in range(4):
    y = limit(y * 10 ** ((TARGET - lufs(y)) / 20), CEIL)
L, tp = lufs(y), true_peak_db(y)
if tp > -1.0:
    y *= 10 ** ((-1.0 - tp - 0.1) / 20)
    L, tp = lufs(y), true_peak_db(y)

wavfile.write(OUT, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
print(f"wrote {OUT}  {DUR:.1f}s  integrated {L:.1f} LUFS  true peak {tp:.2f} dBTP")
