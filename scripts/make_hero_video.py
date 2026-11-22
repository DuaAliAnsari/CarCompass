"""Build a short 16:9 hero.mp4 from the two Corolla photos."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

IMG_DIR = Path(__file__).resolve().parents[1] / "src" / "static" / "img"
OUT = IMG_DIR / "hero.mp4"
W, H = 1280, 720
FPS = 24
SECONDS_EACH = 4
FRAMES_EACH = FPS * SECONDS_EACH
BLEND = 12


def fit_cover(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    scale = max(W / im.width, H / im.height) * 1.18
    nw, nh = int(im.width * scale), int(im.height * scale)
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def ken_burns(im: Image.Image, start=(0.0, 0.0), end=(1.0, 1.0)):
    im = fit_cover(im)
    max_x = max(im.width - W, 1)
    max_y = max(im.height - H, 1)
    frames = []
    for i in range(FRAMES_EACH):
        t = i / (FRAMES_EACH - 1)
        x = int((start[0] + (end[0] - start[0]) * t) * max_x)
        y = int((start[1] + (end[1] - start[1]) * t) * max_y)
        crop = im.crop((x, y, x + W, y + H))
        frames.append(np.asarray(crop))
    return frames


def main():
    imgs = [Image.open(IMG_DIR / "hero-1.png"), Image.open(IMG_DIR / "hero-2.png")]
    frames = []
    frames += ken_burns(imgs[0], (0.0, 0.35), (1.0, 0.0))
    frames += ken_burns(imgs[1], (1.0, 0.2), (0.0, 0.8))

    for i in range(BLEND):
        t = (i + 1) / (BLEND + 1)
        a = frames[FRAMES_EACH - BLEND + i].astype(np.float32)
        b = frames[FRAMES_EACH + i].astype(np.float32)
        frames[FRAMES_EACH - BLEND + i] = (a * (1 - t) + b * t).astype(np.uint8)

    writer = imageio.get_writer(
        str(OUT),
        fps=FPS,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
