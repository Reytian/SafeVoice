#!/usr/bin/env python3
"""
Generate the SafeVoice macOS app icon.

Creates a professional 1024x1024 icon featuring:
- Deep blue-to-teal gradient background
- A stylized shield shape representing "Safe"
- A microphone silhouette with sound wave arcs inside the shield
- Clean, modern, macOS-native aesthetic

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

SIZE = 1024  # master icon size
# Render at 2x then downscale for anti-aliasing
RENDER_SIZE = SIZE * 2


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG_TOP = (12, 25, 70)         # deep navy blue
BG_BOTTOM = (0, 160, 185)     # vibrant teal/cyan

SHIELD_FILL_TOP = (20, 50, 110)    # dark blue
SHIELD_FILL_BOT = (5, 120, 155)    # teal

MIC_COLOR = (255, 255, 255)


# ---------------------------------------------------------------------------
# Helper: fast vertical linear gradient
# ---------------------------------------------------------------------------
def make_gradient_fast(size: int, color_top: tuple, color_bottom: tuple) -> Image.Image:
    """Fast vertical gradient using line drawing."""
    img = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(size - 1, 1)
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * t)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * t)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * t)
        draw.line([(0, y), (size - 1, y)], fill=(r, g, b, 255))
    return img


# ---------------------------------------------------------------------------
# Helper: bezier interpolation for smooth shield curves
# ---------------------------------------------------------------------------
def cubic_bezier(p0, p1, p2, p3, num_points=50):
    """Return points along a cubic bezier curve."""
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


# ---------------------------------------------------------------------------
# Helper: generate a proper shield polygon using bezier curves
# ---------------------------------------------------------------------------
def shield_polygon(cx: float, cy: float, w: float, h: float) -> list:
    """
    Generate a classic heraldic shield shape using bezier curves.
    The shield has:
    - A flat top with rounded corners
    - Sides that curve inward slightly then taper
    - A pointed bottom
    """
    half_w = w / 2
    half_h = h / 2
    r = half_w * 0.18  # corner radius

    y_top = cy - half_h
    y_bottom = cy + half_h
    x_left = cx - half_w
    x_right = cx + half_w

    points = []

    # Top edge: from top-left-inner to top-right-inner
    tl_inner = (x_left + r, y_top)
    tr_inner = (x_right - r, y_top)

    n_top = 30
    for i in range(n_top + 1):
        t = i / n_top
        x = tl_inner[0] + (tr_inner[0] - tl_inner[0]) * t
        points.append((x, y_top))

    # Top-right rounded corner
    n_corner = 20
    corner_cx_r = x_right - r
    corner_cy_r = y_top + r
    for i in range(1, n_corner + 1):
        angle = -math.pi / 2 + (math.pi / 2) * (i / n_corner)
        x = corner_cx_r + r * math.cos(angle)
        y = corner_cy_r + r * math.sin(angle)
        points.append((x, y))

    # Right side: two-segment bezier for a classic shield shape
    # Upper portion: nearly vertical (keeps width)
    # Lower portion: curves inward to the point
    mid_y = cy + half_h * 0.20  # the "waist" where tapering begins

    # Upper right: slight inward curve
    p0 = (x_right, y_top + r)
    p1 = (x_right, cy - half_h * 0.05)
    p2 = (x_right - half_w * 0.02, cy + half_h * 0.05)
    p3 = (x_right - half_w * 0.04, mid_y)
    upper_right = cubic_bezier(p0, p1, p2, p3, num_points=30)
    points.extend(upper_right[1:])

    # Lower right: sweeps inward to bottom point
    p0 = (x_right - half_w * 0.04, mid_y)
    p1 = (x_right - half_w * 0.08, cy + half_h * 0.50)
    p2 = (cx + half_w * 0.18, cy + half_h * 0.82)
    p3 = (cx, y_bottom)
    lower_right = cubic_bezier(p0, p1, p2, p3, num_points=40)
    points.extend(lower_right[1:])

    # Lower left: mirror of lower right
    p0_l = (cx, y_bottom)
    p1_l = (cx - half_w * 0.18, cy + half_h * 0.82)
    p2_l = (x_left + half_w * 0.08, cy + half_h * 0.50)
    p3_l = (x_left + half_w * 0.04, mid_y)
    lower_left = cubic_bezier(p0_l, p1_l, p2_l, p3_l, num_points=40)
    points.extend(lower_left[1:])

    # Upper left: mirror of upper right
    p0_l2 = (x_left + half_w * 0.04, mid_y)
    p1_l2 = (x_left + half_w * 0.02, cy + half_h * 0.05)
    p2_l2 = (x_left, cy - half_h * 0.05)
    p3_l2 = (x_left, y_top + r)
    upper_left = cubic_bezier(p0_l2, p1_l2, p2_l2, p3_l2, num_points=30)
    points.extend(upper_left[1:])

    # Top-left rounded corner
    corner_cx_l = x_left + r
    corner_cy_l = y_top + r
    for i in range(1, n_corner + 1):
        angle = math.pi + (math.pi / 2) * (i / n_corner)
        x = corner_cx_l + r * math.cos(angle)
        y = corner_cy_l + r * math.sin(angle)
        points.append((x, y))

    return points


# ---------------------------------------------------------------------------
# Helper: draw a microphone silhouette
# ---------------------------------------------------------------------------
def draw_microphone(draw: ImageDraw.Draw, cx: float, cy: float, s: float, color: tuple):
    """
    Draw a modern microphone icon.
    s = scale factor relative to RENDER_SIZE.
    """
    # Mic capsule (pill shape)
    cap_w = 72 * s
    cap_h = 130 * s
    cap_r = cap_w / 2
    cap_top = cy - 75 * s
    cap_bot = cap_top + cap_h

    draw.rounded_rectangle(
        [cx - cap_w / 2, cap_top, cx + cap_w / 2, cap_bot],
        radius=cap_r,
        fill=color,
    )

    # Horizontal lines on capsule (mic grille detail)
    grille_color = (SHIELD_FILL_TOP[0], SHIELD_FILL_TOP[1], SHIELD_FILL_TOP[2], 80)
    line_spacing = 16 * s
    line_start_y = cap_top + cap_r + 4 * s
    line_end_y = cap_bot - cap_r - 4 * s
    lw = max(int(2.5 * s), 1)

    # We draw grille lines on a separate layer clipped to the capsule
    grille_layer = Image.new("RGBA", draw.im.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(grille_layer)
    y = line_start_y
    while y < line_end_y:
        # Calculate the width at this y position (it's a pill shape)
        # Distance from center of pill vertically
        dist_from_center = abs(y - (cap_top + cap_bot) / 2)
        if dist_from_center < cap_h / 2 - cap_r:
            line_hw = cap_w / 2 - 8 * s
        else:
            # In the rounded part, calculate chord width
            dy = dist_from_center - (cap_h / 2 - cap_r)
            if dy < cap_r:
                line_hw = math.sqrt(max(cap_r**2 - dy**2, 0)) - 8 * s
            else:
                line_hw = 0
        if line_hw > 0:
            gd.line([(cx - line_hw, y), (cx + line_hw, y)], fill=grille_color, width=lw)
        y += line_spacing

    return grille_layer

    # Cradle (U-shape around capsule bottom)
    cradle_margin = 20 * s
    cradle_lw = max(int(9 * s), 2)
    cradle_box = [
        cx - cap_w / 2 - cradle_margin,
        cap_top + cap_h * 0.2,
        cx + cap_w / 2 + cradle_margin,
        cap_bot + cradle_margin * 1.2,
    ]
    draw.arc(cradle_box, start=0, end=180, fill=color, width=cradle_lw)

    # Stem
    stem_w = 14 * s
    stem_top = (cradle_box[1] + cradle_box[3]) / 2
    stem_bot = cy + 85 * s
    draw.rounded_rectangle(
        [cx - stem_w / 2, stem_top, cx + stem_w / 2, stem_bot],
        radius=stem_w / 2,
        fill=color,
    )

    # Base
    base_w = 72 * s
    base_h = 14 * s
    draw.rounded_rectangle(
        [cx - base_w / 2, stem_bot - 2 * s, cx + base_w / 2, stem_bot + base_h],
        radius=base_h / 2,
        fill=color,
    )


def draw_mic_complete(img: Image.Image, cx: float, cy: float, s: float, color: tuple):
    """Draw the full microphone assembly onto img."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Mic capsule (pill shape)
    cap_w = 72 * s
    cap_h = 130 * s
    cap_r = cap_w / 2
    cap_top = cy - 75 * s
    cap_bot = cap_top + cap_h

    draw.rounded_rectangle(
        [cx - cap_w / 2, cap_top, cx + cap_w / 2, cap_bot],
        radius=cap_r,
        fill=color,
    )

    # Cradle (U-shape around capsule bottom half)
    cradle_margin = 22 * s
    cradle_lw = max(int(9 * s), 2)
    cradle_box = [
        cx - cap_w / 2 - cradle_margin,
        cap_top + cap_h * 0.18,
        cx + cap_w / 2 + cradle_margin,
        cap_bot + cradle_margin * 1.4,
    ]
    draw.arc(cradle_box, start=0, end=180, fill=color, width=cradle_lw)

    # Stem from bottom of cradle arc to base
    stem_w = 14 * s
    cradle_mid_y = (cradle_box[1] + cradle_box[3]) / 2
    stem_bot = cy + 95 * s
    draw.rounded_rectangle(
        [cx - stem_w / 2, cradle_mid_y, cx + stem_w / 2, stem_bot],
        radius=stem_w / 2,
        fill=color,
    )

    # Base
    base_w = 74 * s
    base_h = 14 * s
    draw.rounded_rectangle(
        [cx - base_w / 2, stem_bot - 3 * s, cx + base_w / 2, stem_bot + base_h],
        radius=base_h / 2,
        fill=color,
    )

    img.alpha_composite(layer)

    # Now draw grille lines (clipped to capsule)
    grille_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(grille_layer)
    grille_alpha = 60
    grille_color_rgba = (SHIELD_FILL_TOP[0], SHIELD_FILL_TOP[1], SHIELD_FILL_TOP[2], grille_alpha)
    line_spacing = 17 * s
    lw = max(int(2.5 * s), 1)

    y = cap_top + cap_r
    while y < cap_bot - cap_r:
        half = cap_w / 2 - 10 * s
        if half > 0:
            gd.line([(cx - half, y), (cx + half, y)], fill=grille_color_rgba, width=lw)
        y += line_spacing

    img.alpha_composite(grille_layer)


# ---------------------------------------------------------------------------
# Helper: draw sound wave arcs (clipped inside shield)
# ---------------------------------------------------------------------------
def draw_sound_waves(img: Image.Image, cx: float, cy: float, s: float, shield_mask: Image.Image):
    """Draw concentric arc pairs emanating from mic head, clipped to shield."""
    mic_head_cy = cy - 20 * s

    radii = [105 * s, 148 * s, 190 * s]
    arc_span = 32
    line_widths = [int(10 * s), int(8 * s), int(6 * s)]
    alphas = [180, 130, 80]

    for i, (radius, lw) in enumerate(zip(radii, line_widths)):
        color = (255, 255, 255, alphas[i])

        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)

        box = [
            cx - radius,
            mic_head_cy - radius,
            cx + radius,
            mic_head_cy + radius,
        ]

        # Right arcs
        ld.arc(box, start=-arc_span, end=arc_span, fill=color, width=lw)
        # Left arcs
        ld.arc(box, start=180 - arc_span, end=180 + arc_span, fill=color, width=lw)

        # Clip to shield
        layer.putalpha(
            Image.composite(layer.getchannel("A"), Image.new("L", img.size, 0), shield_mask)
        )

        img.alpha_composite(layer)


# ---------------------------------------------------------------------------
# Main icon generation
# ---------------------------------------------------------------------------
def generate_icon() -> Image.Image:
    """Generate the icon at RENDER_SIZE, then downscale to SIZE for anti-aliasing."""
    RS = RENDER_SIZE

    # 1. Gradient background
    print("  Creating gradient background...")
    icon = make_gradient_fast(RS, BG_TOP, BG_BOTTOM)

    # 2. Subtle vignette (darken edges)
    print("  Adding vignette...")
    dark_layer = Image.new("RGBA", (RS, RS), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dark_layer)
    dd.rectangle([0, 0, RS, RS], fill=(0, 0, 0, 55))
    # Bright center mask
    center_mask = Image.new("L", (RS, RS), 0)
    cm_draw = ImageDraw.Draw(center_mask)
    margin = RS * 0.05
    cm_draw.ellipse([margin, margin, RS - margin, RS - margin], fill=255)
    center_mask = center_mask.filter(ImageFilter.GaussianBlur(radius=RS * 0.12))
    dark_layer.putalpha(
        Image.eval(center_mask, lambda x: int(55 * (1 - x / 255)))
    )
    icon.alpha_composite(dark_layer)

    # 3. Shield
    print("  Drawing shield...")
    shield_cx = RS / 2
    shield_cy = RS / 2 + RS * 0.02  # slight downward shift
    shield_w = RS * 0.56
    shield_h = RS * 0.62

    poly = shield_polygon(shield_cx, shield_cy, shield_w, shield_h)

    # Shield mask for clipping
    shield_mask = Image.new("L", (RS, RS), 0)
    sm_draw = ImageDraw.Draw(shield_mask)
    sm_draw.polygon(poly, fill=255)

    # Shield fill with gradient
    shield_grad = make_gradient_fast(RS, SHIELD_FILL_TOP, SHIELD_FILL_BOT)
    icon.paste(shield_grad, mask=shield_mask)

    # Shield border: draw a crisp white border
    print("  Drawing shield border...")
    border_layer = Image.new("RGBA", (RS, RS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_layer)

    # Multiple passes for a soft glow + crisp edge
    for expand, alpha, width in [(12, 10, 4), (8, 18, 3), (4, 35, 2), (0, 100, 3)]:
        if expand > 0:
            expanded_poly = shield_polygon(shield_cx, shield_cy, shield_w + expand, shield_h + expand)
        else:
            expanded_poly = poly
        el = Image.new("RGBA", (RS, RS), (0, 0, 0, 0))
        ed = ImageDraw.Draw(el)
        ed.polygon(expanded_poly, outline=(200, 230, 255, alpha), width=width)
        if expand > 0:
            el = el.filter(ImageFilter.GaussianBlur(radius=expand // 2))
        border_layer.alpha_composite(el)

    icon.alpha_composite(border_layer)

    # Shield inner highlight (glossy top)
    print("  Adding shield highlight...")
    hl_layer = Image.new("RGBA", (RS, RS), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl_layer)
    hl_w = shield_w * 0.7
    hl_h = shield_h * 0.22
    hl_cy = shield_cy - shield_h * 0.30
    hd.ellipse(
        [shield_cx - hl_w / 2, hl_cy - hl_h / 2, shield_cx + hl_w / 2, hl_cy + hl_h / 2],
        fill=(255, 255, 255, 25),
    )
    hl_layer = hl_layer.filter(ImageFilter.GaussianBlur(radius=RS * 0.04))
    # Clip to shield
    hl_layer.putalpha(
        Image.composite(hl_layer.getchannel("A"), Image.new("L", (RS, RS), 0), shield_mask)
    )
    icon.alpha_composite(hl_layer)

    # 4. Microphone
    print("  Drawing microphone...")
    mic_scale = RS / 512  # scale factor
    mic_cy = shield_cy + RS * 0.02
    draw_mic_complete(icon, shield_cx, mic_cy, mic_scale, MIC_COLOR)

    # 5. Sound wave arcs (clipped to shield)
    print("  Drawing sound waves...")
    draw_sound_waves(icon, shield_cx, mic_cy, mic_scale, shield_mask)

    # 6. Outer glow around shield
    print("  Adding outer glow...")
    glow = Image.new("RGBA", (RS, RS), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    glow_poly = shield_polygon(shield_cx, shield_cy, shield_w + 40, shield_h + 40)
    gd.polygon(glow_poly, fill=(80, 200, 255, 18))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=25))
    # Exclude interior
    outer_mask = Image.new("L", (RS, RS), 255)
    om_draw = ImageDraw.Draw(outer_mask)
    om_draw.polygon(poly, fill=0)
    glow.putalpha(
        Image.composite(glow.getchannel("A"), Image.new("L", (RS, RS), 0), outer_mask)
    )
    icon.alpha_composite(glow)

    # 7. Downscale for anti-aliasing
    print("  Downscaling for anti-aliasing...")
    icon = icon.resize((SIZE, SIZE), Image.LANCZOS)

    return icon


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
