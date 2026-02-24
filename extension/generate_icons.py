"""
KAVACH.AI Extension — Icon Generator (Fixed)
Fixes bug: previously had b"\\x89PNG" (literal backslash) instead of b"\x89PNG" (actual byte).
Run: py extension/generate_icons.py
"""

import os
import struct
import zlib

ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
os.makedirs(ICONS_DIR, exist_ok=True)

SIZES    = [16, 48, 128]
BG       = (15, 23, 42)       # dark navy
ACCENT   = (0, 243, 255)      # neon cyan (KAVACH.AI brand)


def make_png(size: int) -> bytes:
    w = h = size
    pixels = [BG] * (w * h)

    # Draw a shield silhouette
    cx, cy = w // 2, int(h * 0.42)
    r = max(4, int(w * 0.32))

    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            in_top    = dy <= 0 and dx * dx + dy * dy <= r * r
            half_w    = r - dy
            in_bottom = 0 < dy <= r and abs(dx) <= half_w
            if in_top or in_bottom:
                pixels[y * w + x] = ACCENT

    # ── PNG structure ─────────────────────────────────────────
    def chunk(name: bytes, data: bytes) -> bytes:
        body = name + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB

    raw = bytearray()
    for y in range(h):
        raw.append(0)                           # no filter
        for x in range(w):
            raw.extend(pixels[y * w + x])       # R G B

    idat = zlib.compress(bytes(raw), 9)

    # FIX: use actual bytes b"\x89PNG\r\n\x1a\n" (not escaped string)
    png  = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", idat)
    png += chunk(b"IEND", b"")
    return png


if __name__ == "__main__":
    for size in SIZES:
        path = os.path.join(ICONS_DIR, f"icon{size}.png")
        with open(path, "wb") as f:
            f.write(make_png(size))
        print(f"  ✓ {path}")
    print("Done — icons ready, reload extension in Chrome.")
