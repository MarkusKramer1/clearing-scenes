#!/usr/bin/env python3
"""Render one GIF per scene for the README: a turn around it, then a pass through.

Frames come out of `clearing.render`, which is plain numpy -- no display, no
browser, no OpenGL. `ffmpeg` turns them into a GIF via a per-clip palette,
which is the difference between 2 MB and 12 MB for the same picture; without
ffmpeg on PATH it falls back to Pillow.

    python scripts/make_gifs.py [--scenes christ-church] [--width 720]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from clearing import render, scene as scene_mod  # noqa: E402


def caption(img: np.ndarray, text: str) -> np.ndarray:
    """A 5x7 bitmap caption, bottom left. Small enough to keep in this file."""
    font = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
        "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
        "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
        "H": ["10001", "10001", "11111", "10001", "10001", "10001", "10001"],
        "I": ["111", "010", "010", "010", "010", "010", "111"],
        "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
        "K": ["10001", "10010", "11100", "10010", "10001", "10001", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
        "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["001", "011", "101", "001", "001", "001", "111"],
        "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "01110", "10001", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        "-": ["000", "000", "000", "111", "000", "000", "000"],
        ",": ["00", "00", "00", "00", "00", "01", "10"],
        ".": ["0", "0", "0", "0", "0", "0", "1"],
        " ": ["0", "0", "0", "0", "0", "0", "0"],
    }
    scale, x0 = 2, 10
    y0 = img.shape[0] - 7 * scale - 10
    x = x0
    for ch in text.upper():
        glyph = font.get(ch, font[" "])
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    ys, xs = y0 + r * scale, x + c * scale
                    img[ys:ys + scale, xs:xs + scale] = (236, 238, 242)
        x += (len(glyph[0]) + 1) * scale
    return img


def render_scene(name: str, width: int, height: int, orbit: int, fly: int,
                 fps: int, out: Path) -> None:
    s = scene_mod.load(name, with_cloud=True)
    cams = render.orbit_and_fly(s, n_orbit=orbit, n_fly=fly)
    label = (f"{s.title}  {s.n_nodes} vertices  "
             f"{s.doc['surface']['area_m2']:,.0f} m2").replace(",", " ")

    tmp = Path(tempfile.mkdtemp(prefix=f"gif-{name}-"))
    try:
        from PIL import Image
        for k, cam in enumerate(cams):
            img = render.draw_scene(s, cam, width, height)
            Image.fromarray(caption(img, label)).save(tmp / f"{k:04d}.png")
            if (k + 1) % 20 == 0:
                print(f"    {k + 1}/{len(cams)} frames", flush=True)

        if shutil.which("ffmpeg"):
            pal = tmp / "palette.png"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", str(tmp / "%04d.png"),
                 "-vf", "palettegen=max_colors=128:stats_mode=diff", str(pal)],
                check=True)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", str(tmp / "%04d.png"), "-i", str(pal),
                 "-lavfi", "paletteuse=dither=none",
                 "-loop", "0", str(out)],
                check=True)
        else:
            frames = [Image.open(p) for p in sorted(tmp.glob("[0-9]*.png"))]
            frames[0].save(out, save_all=True, append_images=frames[1:],
                           duration=int(1000 / fps), loop=0, optimize=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=380)
    ap.add_argument("--orbit", type=int, default=34)
    ap.add_argument("--fly", type=int, default=22)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()

    out_dir = REPO / "docs" / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (a.scenes or scene_mod.available()):
        out = out_dir / f"{name}.gif"
        print(f"[gif] {name}", flush=True)
        render_scene(name, a.width, a.height, a.orbit, a.fly, a.fps, out)
        print(f"      {out.relative_to(REPO)}  "
              f"{out.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
