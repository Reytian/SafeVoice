#!/usr/bin/env python3
"""
Generate the SafeVoice macOS app icon ("Bare Wave" mark).

Design: five staggered organic waveform bars in warm ink on a warm-paper
macOS squircle. Monochrome and flat in the Typeless tradition: no
gradients, no security iconography, nothing but the voice. The 16 px
render is an optical variant (middle three bars, enlarged) because five
bars smear below menubar size.

Outputs:
- assets/icon_1024.png (1024x1024 source)
- assets/SafeVoice.iconset/ (all required sizes)
- assets/SafeVoice.icns (macOS icon file)
"""

import math
import os
import subprocess
import sys

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
# Palette: warm paper + warm ink (flat, monochrome)
# ---------------------------------------------------------------------------
PAPER = (247, 244, 238, 255)   # warm off-white
INK = (26, 24, 22, 255)        # warm near-black

# macOS Big Sur+ icon grid: the squircle occupies ~824/1024 of the canvas,
# leaving transparent margin so the Dock shows a rounded shape, not a square.
SQUIRCLE_FRACTION = 824 / 1024
SQUIRCLE_EXPONENT = 5.0       # superellipse "squareness" (Apple-like)

# Glyph geometry lives in a 0..150 design grid mapped onto the squircle box,
# mirroring the approved SVG concept so preview and asset stay identical.
DESIGN_GRID = 150.0

# Organic waveform bars: (x, y, w, h) in the design grid. Widths, heights,
# and vertical centers are deliberately uneven; perfect symmetry reads
# mechanical, speech does not.
BARS = [
    (36.5, 61, 13, 32),
    (52.5, 49, 13, 54),
    (68.0, 38, 14, 74),
    (84.5, 53, 13, 48),
    (100.5, 60, 13, 34),
]

# 16 px optical variant: three bars with exaggerated spacing. At 16 px the
# squircle is ~13 px wide, so gaps must stay >= 1 px after scaling; zooming
# the master bars closes the gaps and reads as a blob. Designed gap-first.
BARS_TINY = [
    (23.0, 50, 23, 50),
    (63.5, 32, 23, 86),
    (104.0, 46, 23, 58),
]


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
# Glyph: five staggered organic waveform bars
# ---------------------------------------------------------------------------
def glyph_mask(size: int, box_origin: float, box_size: float,
               simplified: bool = False) -> Image.Image:
    """L-mode mask of the Bare Wave mark.

    With ``simplified=True`` only the middle three bars render, enlarged:
    the optical variant for 16 px, where five bars smear together.
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    scale = box_size / DESIGN_GRID
    zoom = 1.0  # BARS_TINY is already gap-first; no zoom for either variant
    grid_center = DESIGN_GRID / 2.0
    bars = BARS_TINY if simplified else BARS

    def t(v: float) -> float:  # design-grid coordinate -> canvas
        zoomed = grid_center + (v - grid_center) * zoom
        return box_origin + zoomed * scale

    def s(v: float) -> float:  # design-grid length -> canvas
        return v * scale * zoom

    for bx, by, bw, bh in bars:
        draw.rounded_rectangle(
            [t(bx), t(by), t(bx + bw), t(by + bh)], radius=s(bw) / 2.0, fill=255,
        )

    return mask


# ---------------------------------------------------------------------------
# Icon assembly
# ---------------------------------------------------------------------------
def generate_icon(simplified: bool = False) -> Image.Image:
    size = RENDER_SIZE
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    box_size = round(size * SQUIRCLE_FRACTION)
    box_origin = (size - box_size) / 2.0
    sq_mask = squircle_mask(size, box_size)

    # Soft baked shadow under the squircle (subtle, downward).
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 45))
    shadow.paste(shadow_layer, (0, round(size * 0.012)), sq_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.012))
    canvas = Image.alpha_composite(canvas, shadow)

    # Flat warm-paper squircle
    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    plate.paste(Image.new("RGBA", (size, size), PAPER), (0, 0), sq_mask)
    canvas = Image.alpha_composite(canvas, plate)

    # Ink glyph
    g_mask = glyph_mask(size, box_origin, box_size, simplified=simplified)
    glyph = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glyph.paste(Image.new("RGBA", (size, size), INK), (0, 0), g_mask)
    canvas = Image.alpha_composite(canvas, glyph)

    return canvas.resize((SIZE, SIZE), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Generate all sizes and .icns
# ---------------------------------------------------------------------------
def main():
    print("Generating SafeVoice icon (1024x1024)...")
    icon = generate_icon()
    # Optical variant for the tiniest size: three bars, enlarged.
    icon_tiny = generate_icon(simplified=True)

    # Save master 1024 PNG
    master_path = os.path.join(ASSETS_DIR, "icon_1024.png")
    icon.save(master_path, "PNG")
    print(f"  Saved: {master_path}")

    # Generate iconset PNGs
    print("Generating iconset PNGs...")
    iconset_specs = [
        ("icon_16x16.png", 16, True),
        ("icon_16x16@2x.png", 32, False),
        ("icon_32x32.png", 32, False),
        ("icon_32x32@2x.png", 64, False),
        ("icon_128x128.png", 128, False),
        ("icon_128x128@2x.png", 256, False),
        ("icon_256x256.png", 256, False),
        ("icon_256x256@2x.png", 512, False),
        ("icon_512x512.png", 512, False),
        ("icon_512x512@2x.png", 1024, False),
    ]

    for filename, px, use_tiny in iconset_specs:
        source = icon_tiny if use_tiny else icon
        resized = source.resize((px, px), Image.LANCZOS)
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
