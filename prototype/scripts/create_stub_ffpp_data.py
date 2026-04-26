from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path
from typing import Callable, Iterable, Tuple


Rgb = Tuple[int, int, int]


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_png(path: Path, width: int, height: int, pixel_fn: Callable[[int, int], Rgb]) -> None:
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            rows.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=9)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(png_chunk(b"IHDR", ihdr))
    png.extend(png_chunk(b"IDAT", idat))
    png.extend(png_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def real_pattern(seed: int) -> Callable[[int, int], Rgb]:
    def pixel(x: int, y: int) -> Rgb:
        r = (40 + x + seed * 17) % 256
        g = (90 + y * 2 + seed * 23) % 256
        b = (130 + (x + y) // 2 + seed * 11) % 256
        return r, g, b

    return pixel


def fake_pattern(seed: int) -> Callable[[int, int], Rgb]:
    def pixel(x: int, y: int) -> Rgb:
        block = ((x // 8) + (y // 8) + seed) % 2
        edge = 255 if x % 16 in (0, 15) or y % 16 in (0, 15) else 0
        base = 180 if block else 30
        r = min(255, base + edge)
        g = (70 + seed * 31 + x * 3) % 256
        b = (180 + y * 2 + seed * 29) % 256
        return r, g, b

    return pixel


def generate_images(target_dir: Path, prefix: str, count: int, pattern_factory: Callable[[int], Callable[[int, int], Rgb]]) -> Iterable[Path]:
    paths = []
    for idx in range(count):
        path = target_dir / f"{prefix}_{idx:02d}.png"
        write_png(path, width=96, height=96, pixel_fn=pattern_factory(idx))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic FF++-style stub dataset for smoke testing.")
    parser.add_argument("--real-dir", required=True, help="Target directory for stub real images.")
    parser.add_argument("--fake-dir", required=True, help="Target directory for stub fake images.")
    parser.add_argument("--num-real", type=int, default=4, help="Number of real stub images to create.")
    parser.add_argument("--num-fake", type=int, default=4, help="Number of fake stub images to create.")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    real_paths = list(generate_images(real_dir, "real_stub", args.num_real, real_pattern))
    fake_paths = list(generate_images(fake_dir, "fake_stub", args.num_fake, fake_pattern))

    print(f"Created {len(real_paths)} real stub images in {real_dir}")
    print(f"Created {len(fake_paths)} fake stub images in {fake_dir}")


if __name__ == "__main__":
    main()
