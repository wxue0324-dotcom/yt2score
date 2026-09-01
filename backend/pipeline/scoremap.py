"""Where each bar sits on the engraved page, so playback can be followed on it.

MuseScore's SVG carries geometry but no timing: a note is a `<path class="Note">`
at some coordinate with nothing to say when it sounds. Matching those paths back
to notes would mean re-deriving the layout, so this takes the cheaper route and
maps *bars* instead. Bar lines are unambiguous in the SVG, and at a fixed tempo
a bar's start time is arithmetic — which is enough to put a cursor within a
fraction of a beat of the right place, interpolating across the bar.

The geometry is quietly redundant. Every bar line is drawn twice: one stroke
from the top stave down to the lower stave's top, and a second covering the
lower stave. Both share an x, so the pair also delimits the system vertically —
which is how systems are found here, rather than by clustering stave lines and
guessing at the gap that separates one system from the next.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_VIEWBOX = re.compile(r'viewBox="([^"]+)"')
_BARLINE = re.compile(r'<polyline class="BarLine"[^>]*points="([^"]+)"')

# Two strokes closer together than this are one double bar line, not two bars.
# Measured on a final bar line, the thin and thick strokes sit 61 units apart on
# a 9924-unit page, against roughly 1300 for the narrowest real bar — so the
# threshold has a wide margin either side. Scaled to the page so it survives a
# different paper size, with a floor for very narrow pages.
_DOUBLE_BAR = 0.012

# Bar lines within this much of each other vertically belong to the same run.
_ROW_TOLERANCE = 30.0


def _bar_lines(svg: str) -> list[tuple[float, float, float]]:
    out = []
    for match in _BARLINE.finditer(svg):
        points = [tuple(map(float, p.split(","))) for p in match.group(1).split()]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        out.append((sum(xs) / len(xs), min(ys), max(ys)))
    out.sort(key=lambda b: (b[1], b[0]))
    return out


def _systems(svg: str, page_width: float) -> list[dict]:
    """Systems on one page, each with the x of every bar line across it."""
    bars = _bar_lines(svg)
    if not bars:
        return []

    rows: list[list[tuple[float, float, float]]] = []
    for bar in bars:
        if rows and abs(bar[1] - rows[-1][0][1]) < _ROW_TOLERANCE:
            rows[-1].append(bar)
        else:
            rows.append([bar])

    gap = max(_ROW_TOLERANCE * 2, page_width * _DOUBLE_BAR)
    systems = []
    index = 0
    while index < len(rows):
        # One system's strokes run head to tail down its staves: each row starts
        # where the row above it ended. Follow that chain rather than assuming a
        # fixed number of staves — a grand staff gives two rows, but a score
        # with a melody line above the piano gives three, and pairing them off
        # two at a time counts every system twice.
        first = index
        while (index + 1 < len(rows)
               and abs(rows[index + 1][0][1] - rows[index][0][2]) < _ROW_TOLERANCE):
            index += 1
        edges: list[float] = []
        for x in sorted({round(b[0], 2) for b in rows[first]}):
            if edges and x - edges[-1] < gap:
                edges[-1] = x
            else:
                edges.append(x)
        if len(edges) >= 2:
            systems.append({
                "top": rows[first][0][1],
                "bottom": rows[index][0][2],
                "edges": edges,
            })
        index += 1
    return systems


def build(svg_pages: list[Path], beats_per_bar: int, tempo: float) -> dict | None:
    """A time-to-position map for the engraved pages, or None if unusable.

    Bars are numbered in reading order across the whole score, so a bar index
    is also its position in time: bar *n* starts at `n * beats_per_bar` beats.
    """
    if not svg_pages or tempo <= 0:
        return None

    pages = []
    total = 0
    for path in svg_pages:
        try:
            svg = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        box = _VIEWBOX.search(svg)
        if not box:
            continue
        _, _, width, height = (float(v) for v in box.group(1).split())
        systems = []
        for system in _systems(svg, width):
            edges = system["edges"]
            bars = []
            for left, right in zip(edges, edges[1:]):
                bars.append({"i": total, "x0": left, "x1": right})
                total += 1
            if bars:
                systems.append({"top": system["top"], "bottom": system["bottom"],
                                "bars": bars})
        pages.append({"width": width, "height": height, "systems": systems})

    if not total:
        return None
    return {
        "tempo": round(float(tempo), 4),
        "beats_per_bar": int(beats_per_bar),
        "bars": total,
        "pages": pages,
    }


def write(svg_pages: list[Path], beats_per_bar: int, tempo: float,
          out_path: Path) -> Path | None:
    data = build(svg_pages, beats_per_bar, tempo)
    if data is None:
        return None
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out_path
