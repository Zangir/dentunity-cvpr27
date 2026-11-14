"""Store panoramic images once at 640 px on the long side (aspect kept, JPEG q=95).

Training and evaluation run at imgsz 640, where Ultralytics resizes the long side
to 640 anyway; pre-resizing only removes the cost of decoding 1615x840 JPEGs on
every mosaic draw. Normalised labels are unchanged. Writes data/px640/<sha256>.jpg.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from .common import DATA, read_json

OUT = DATA / "px640"


def one(r):
    dst = OUT / f"{r['sha256']}.jpg"
    if dst.exists():
        return
    im = Image.open(r["path"]).convert("RGB")
    w, h = im.size
    s = 640 / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    im.save(dst, quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = read_json(DATA / "bench.json")["px"]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, px))
    print("resized", len(list(OUT.iterdir())))


if __name__ == "__main__":
    main()
