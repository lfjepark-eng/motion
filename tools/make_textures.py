"""Paper textures for the four layers. Seeded, so every run writes identical files."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

W, H = 1920, 1080
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def value_noise(rng, scale, octaves=4):
    """Smooth fractal noise in [-1, 1] built from upsampled random grids."""
    acc = np.zeros((H, W), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gw, gh = max(2, int(W / scale * 2**o)), max(2, int(H / scale * 2**o))
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        img = Image.fromarray(grid).resize((W, H), Image.BICUBIC)
        acc += np.asarray(img) * amp
        total += amp
        amp *= 0.5
    acc /= total
    return acc / (np.abs(acc).max() + 1e-6)


def vignette(strength):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
    return 1.0 - strength * d**2


def fibres(rng, count, length, width, alpha, blur):
    """Long thin kozo fibres as a darkening/lightening mask in [-1, 1]."""
    layer = Image.new("L", (W, H), 128)
    draw = ImageDraw.Draw(layer)
    for _ in range(count):
        x, y = rng.uniform(-100, W + 100), rng.uniform(-100, H + 100)
        ang = rng.uniform(0, np.pi)
        seg = rng.integers(4, 9)
        pts = [(x, y)]
        for _ in range(seg):
            ang += rng.normal(0, 0.35)
            step = length / seg * rng.uniform(0.6, 1.4)
            x, y = x + np.cos(ang) * step, y + np.sin(ang) * step
            pts.append((x, y))
        tone = 128 + int(rng.choice([-1, 1]) * alpha * rng.uniform(0.4, 1.0))
        draw.line(pts, fill=tone, width=int(rng.integers(1, width + 1)))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return (np.asarray(layer, np.float32) - 128) / 128


def save(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(OUT / name, quality=90)


# L1 한지: cool-white sheet, long visible fibres, soft cloudy density.
rng = np.random.default_rng(1031)
base = hex_rgb("#ECE9E1")
cloud = value_noise(rng, 260, 5)
fib = fibres(rng, 1400, 260, 3, 70, 0.8) + 0.6 * fibres(rng, 3000, 90, 1, 50, 0.4)
lum = 1 + 0.025 * cloud + 0.06 * fib
img = base[None, None, :] * lum[..., None] * vignette(0.06)[..., None]
save(img, "hanji.jpg")

# L2 크라프트: brown with dark specks and short fibres.
rng = np.random.default_rng(2031)
base = hex_rgb("#B4875A")
cloud = value_noise(rng, 180, 5)
fib = fibres(rng, 5000, 40, 1, 60, 0.5)
specks = (rng.random((H, W)) > 0.99965).astype(np.float32)
specks = gaussian_filter(specks, 1.4) * 9
lum = 1 + 0.05 * cloud + 0.09 * fib - 0.35 * specks
img = base[None, None, :] * lum[..., None] * vignette(0.06)[..., None]
save(img, "kraft.jpg")

# L3 신문지: grey newsprint, fine grain, faint show-through.
rng = np.random.default_rng(3031)
base = hex_rgb("#D8D4CB")
cloud = value_noise(rng, 300, 4)
grain = rng.standard_normal((H, W)).astype(np.float32)
lum = 1 + 0.03 * cloud + 0.03 * grain
img = base[None, None, :] * lum[..., None] * vignette(0.06)[..., None]
save(img, "newsprint.jpg")

# L4 포스터 grain: neutral mid-grey overlay used with mix-blend-mode: overlay.
rng = np.random.default_rng(4031)
grain = rng.standard_normal((H, W)).astype(np.float32)
grain = gaussian_filter(grain, 0.6) * 2.2
save(np.repeat((128 + 18 * grain)[..., None], 3, axis=2), "poster-grain.jpg")

print("wrote", sorted(p.name for p in OUT.iterdir()))
