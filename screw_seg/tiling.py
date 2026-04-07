from __future__ import annotations

from .structures import TileWindow


def generate_tiles(width: int, height: int, tile_size: int, overlap_ratio: float) -> list[TileWindow]:
    step = max(1, int(tile_size * (1.0 - overlap_ratio)))
    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    windows: list[TileWindow] = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            windows.append(TileWindow(x1=x1, y1=y1, x2=x2, y2=y2))
    return windows
