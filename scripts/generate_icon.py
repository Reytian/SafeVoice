#!/usr/bin/env python3
"""
Generate the SafeVoice macOS app icon ("Lockwave" mark).

Design: a single white padlock whose keyhole is a three-bar waveform,
centered on a macOS squircle (superellipse) filled with a diagonal
deep-navy -> cyan gradient. One glyph, one gradient, no ornament: the
lock silhouette stays legible at 16 px (menubar) while the waveform
keyhole reads at Dock sizes and above.

Outputs:
- assets/icon_1024.png (1024x1024 source)
- assets/SafeVoice.iconset/ (all required sizes)
- assets/SafeVoice.icns (macOS icon file)
"""

import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
ICONSET_DIR = os.path.join(ASSETS_DIR, "SafeVoice.iconset")

os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(ICONSET_DIR, exist_ok=True)

SIZE = 1024      # master icon size
RENDER_SIZE = SIZE * 2  # render at 2x, downscale for anti-aliasing

# ---------------------------------------------------------------------------
# Palette (diagonal gradient, top-left -> bottom-right)
# ---------------------------------------------------------------------------
GRAD_START = (10, 26, 82)     # deep indigo-navy
GRAD_MID = (14, 79, 168)      # royal blue
GRAD_END = (0, 194, 209)      # cyan
GRAD_MID_POS = 0.55           # where the mid stop sits along the diagonal

GLYPH_COLOR = (255, 255, 255, 255)

# macOS Big Sur+ icon grid: the squircle occupies ~824/1024 of the canvas,
# leaving transparent margin so the Dock shows a rounded shape, not a square.
SQUIRCLE_FRACTION = 824 / 1024
SQUIRCLE_EXPONENT = 5.0       # superellipse "squareness" (Apple-like)

# Glyph geometry in a 0..150 design grid, mapped onto the squircle box.
# Mirrors the approved SVG concept so preview and asset stay identical.
DESIGN_GRID = 150.0


# ---------------------------------------------------------------------------
# Background: diagonal three-stop gradient
# ---------------------------------------------------------------------------
def make_diagonal_gradient(size: int) -> Image.Image:
    """Diagonal gradient with a mid stop, computed vectorized via numpy."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    t = (xx + yy) / 2.0  # 0 at top-left, 1 at bottom-right
    # The squircle is inset from the canvas, so the raw diagonal never
    # reaches its endpoints inside the visible shape; remap so full navy
    # and full cyan both land within the squircle, not in cropped corners.
    t = np.clip((t - 0.08) / 0.78, 0.0, 1.0)

    start = np.array(GRAD_START, dtype=np.float32)
    mid = np.array(GRAD_MID, dtype=np.float32)
    end = np.array(GRAD_END, dtype=np.float32)

    rgb = np.empty((size, size, 3), dtype=np.float32)
    first = t < GRAD_MID_POS
    t1 = (t / GRAD_MID_POS)[..., None]
    t2 = ((t - GRAD_MID_POS) / (1.0 - GRAD_MID_POS))[..., None]
    rgb = np.where(
        first[..., None],
        start + (mid - start) * t1,
        mid + (end - mid) * t2,
    )

    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return Image.fromarray(out, "RGBA")


# ---------------------------------------------------------------------------
# Squircle (superellipse) mask
# ---------------------------------------------------------------------------
def squircle_mask(size: int, box_size: int, n: float = SQUIRCLE_EXPONENT) -> Image.Image:
    """L-mode mask of a centered superellipse |x/a|^n + |y/a|^n = 1."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    a = box_size / 2.0
    cx = cy = size / 2.0
    points = []
    steps = 720
    for i in range(steps):
        theta = 2.0 * math.pi * i / steps
        ct, st = math.cos(theta), math.sin(theta)
        x = cx + a * math.copysign(abs(ct) ** (2.0 / n), ct)
        y = cy + a * math.copysign(abs(st) ** (2.0 / n), st)
        points.append((x, y))
    draw.polygon(points, fill=255)
    return mask


# ---------------------------------------------------------------------------
# Glyph: padlock with a waveform keyhole (punched through)
# ---------------------------------------------------------------------------
def glyph_mask(size: int, box_origin: float, box_size: float) -> Image.Image:
    """L-mode mask of the lock glyph, bars punched out (0) of the body (255)."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    scale = box_size / DESIGN_GRID
    # Slight enlargement around the grid center for optical presence
    # (macOS glyphs typically claim a bit over half the squircle).
    glyph_zoom = 1.06
    grid_center = DESIGN_GRID / 2.0

    def t(v: float) -> float:  # design-grid coordinate -> canvas
        zoomed = grid_center + (v - grid_center) * glyph_zoom
        return box_origin + zoomed * scale

    def s(v: float) -> float:  # design-grid length -> canvas
        return v * scale * glyph_zoom

    # Shackle: upper half-ring centered at (75, 56), radius 20, stroke 13,
    # with straight legs down to y=66 (the body overlaps the leg ends).
    shackle_cx, shackle_cy = t(75), t(56)
    r = s(20)
    stroke = s(13)
    outer = r + stroke / 2.0
    draw.arc(
        [shackle_cx - outer, shackle_cy - outer, shackle_cx + outer, shackle_cy + outer],
        start=180, end=360, fill=255, width=max(1, round(stroke)),
    )
    for leg_x in (t(55), t(95)):
        draw.rectangle(
            [leg_x - stroke / 2.0, shackle_cy, leg_x + stroke / 2.0, t(66)],
            fill=255,
        )

    # Body
    draw.rounded_rectangle(
        [t(37), t(64), t(37 + 76), t(64 + 58)], radius=s(17), fill=255,
    )

    # Waveform keyhole: three bars punched out of the body.
    bars = [  # (x, y, w, h) in design grid
        (56, 84, 8, 18),
        (71, 78, 8, 30),
        (86, 84, 8, 18),
    ]
    for bx, by, bw, bh in bars:
        draw.rounded_rectangle(
            [t(bx), t(by), t(bx + bw), t(by + bh)], radius=s(bw) / 2.0, fill=0,
        )

    return mask


# ---------------------------------------------------------------------------
# Icon assembly
# ---------------------------------------------------------------------------
def generate_icon() -> Image.Image:
    size = RENDER_SIZE
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    box_size = round(size * SQUIRCLE_FRACTION)
    box_origin = (size - box_size) / 2.0
    sq_mask = squircle_mask(size, box_size)

    # Soft baked shadow under the squircle (subtle, downward).
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 60))
    shadow.paste(shadow_layer, (0, round(size * 0.012)), sq_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.012))
    canvas = Image.alpha_composite(canvas, shadow)

    # Gradient squircle
    gradient = make_diagonal_gradient(size)
    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    plate.paste(gradient, (0, 0), sq_mask)
    canvas = Image.alpha_composite(canvas, plate)

    # White glyph with punched waveform
    g_mask = glyph_mask(size, box_origin, box_size)
    glyph = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glyph.paste(Image.new("RGBA", (size, size), GLYPH_COLOR), (0, 0), g_mask)
    canvas = Image.alpha_composite(canvas, glyph)

    return canvas.resize((SIZE, SIZE), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Generate all sizes and .icns
# ---------------------------------------------------------------------------
def main():
    print("Generating SafeVoice icon (1024x1024)...")
    icon = generate_icon()

    # Save master 1024 PNG
    master_path = os.path.join(ASSETS_DIR, "icon_1024.png")
    icon.save(master_path, "PNG")
    print(f"  Saved: {master_path}")

    # Generate iconset PNGs
    print("Generating iconset PNGs...")
    iconset_specs = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]

    for filename, px in iconset_specs:
        resized = icon.resize((px, px), Image.LANCZOS)
        out_path = os.path.join(ICONSET_DIR, filename)
        resized.save(out_path, "PNG")
        print(f"  Saved: {out_path} ({px}x{px})")

    # Build .icns with iconutil
    icns_path = os.path.join(ASSETS_DIR, "SafeVoice.icns")
    print(f"Building {icns_path} with iconutil...")
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", ICONSET_DIR, "-o", icns_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Success: {icns_path}")
    except subprocess.CalledProcessError as e:
        print(f"  iconutil failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("  iconutil not found (not on macOS?). Skipping .icns generation.", file=sys.stderr)

    print("\nDone! Icon files generated in assets/")


if __name__ == "__main__":
    main()
