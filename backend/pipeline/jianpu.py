"""Render a melody line as 簡譜 (numbered notation) SVG.

Notation rules implemented:
  1-7        scale degrees relative to the key's "do"; 0 is a rest
  dot above  one octave up (stacked for two); dot below one octave down
  underline  halves the value (one = quaver, two = semiquaver)
  dot right  dotted note (1.5x)
  dash right adds one beat each (so "1 - -" is three beats)
  # / b      accidental, placed before the digit
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

from .analyze import Analysis
from .quantize import GridNote

# Semitones above "do" -> (degree digit, accidental). Sharps are the usual
# choice in 簡譜 for the raised degrees, flat for the lowered seventh.
_DEGREE = {
    0: ("1", ""), 1: ("1", "#"), 2: ("2", ""), 3: ("3", "b"), 4: ("3", ""),
    5: ("4", ""), 6: ("4", "#"), 7: ("5", ""), 8: ("5", "#"), 9: ("6", ""),
    10: ("7", "b"), 11: ("7", ""),
}

_PC = {"C": 0, "C#": 1, "D": 2, "E-": 3, "D#": 3, "E": 4, "F": 5, "F#": 6,
       "G": 7, "G#": 8, "A": 9, "B-": 10, "A#": 10, "B": 11}

PAGE_W, PAGE_H = 794, 1123          # A4 at 96 dpi
MARGIN_X, MARGIN_TOP = 56, 70
SLOT_W = 34                          # horizontal space for one symbol
LINE_H = 78
BASE_FONT = 26


@dataclass
class Cell:
    """One printed symbol: a digit, a rest, or a continuation dash."""
    text: str                # "1".."7", "0", or "-"
    accidental: str = ""
    octave: int = 0          # +1 = one dot above, -1 = one dot below
    underlines: int = 0
    dotted: bool = False
    is_dash: bool = False


@dataclass
class Measure:
    cells: list[Cell] = field(default_factory=list)

    @property
    def width_slots(self) -> int:
        return max(len(self.cells), 1)


def _duration_shape(ql: float) -> tuple[int, bool, int]:
    """quarter-length -> (underlines, dotted, extra dashes)."""
    table = [
        (0.25, 2, False), (0.375, 2, True), (0.5, 1, False), (0.75, 1, True),
        (1.0, 0, False), (1.5, 0, True),
    ]
    if ql < 1.75:
        _, ul, dot = min(table, key=lambda t: abs(t[0] - ql))
        return ul, dot, 0
    beats = int(round(ql))
    return 0, False, max(0, beats - 1)


def _tonic_pc(analysis: Analysis) -> tuple[int, str]:
    """Return the pitch class of "do" and its display name.

    Minor keys are written La-based: the numbers come from the relative major,
    which is how minor-key 簡譜 is conventionally printed.
    """
    pc = _PC.get(analysis.tonic, 0)
    if analysis.mode == "minor":
        pc = (pc + 3) % 12
    names = ["C", "#C", "D", "bE", "E", "F", "#F", "G", "bA", "A", "bB", "B"]
    return pc, names[pc]


def melody_to_measures(notes: list[GridNote], analysis: Analysis,
                       max_bars: int | None = None) -> list[Measure]:
    """Walk the timeline bar by bar, emitting notes and filling gaps with rests."""
    bar_len = float(analysis.beats_per_bar)
    tonic_pc, _ = _tonic_pc(analysis)

    if not notes:
        return []

    pitches = sorted(n.pitch for n in notes)
    median = pitches[len(pitches) // 2]
    # Reference octave: place "do" just below the median so most of the melody
    # needs no octave dots at all.
    ref = median - ((median - tonic_pc) % 12)

    total = max(n.offset + n.duration for n in notes)
    bar_count = int(total // bar_len) + 1
    if max_bars:
        bar_count = min(bar_count, max_bars)

    ordered = sorted(notes, key=lambda n: (n.offset, -n.pitch))
    measures: list[Measure] = []

    for b in range(bar_count):
        start, end = b * bar_len, (b + 1) * bar_len
        m = Measure()
        cursor = start
        in_bar = [n for n in ordered if start <= n.offset < end]

        for n in in_bar:
            if n.offset > cursor + 1e-6:
                m.cells.extend(_rest_cells(n.offset - cursor))
            dur = min(n.duration, end - n.offset)
            if dur <= 0:
                continue
            ul, dotted, dashes = _duration_shape(dur)
            semis = (n.pitch - tonic_pc) % 12
            digit, acc = _DEGREE[semis]
            octave = (n.pitch - ref) // 12
            m.cells.append(Cell(text=digit, accidental=acc, octave=octave,
                                underlines=ul, dotted=dotted))
            m.cells.extend(Cell(text="-", is_dash=True) for _ in range(dashes))
            cursor = n.offset + dur

        if cursor < end - 1e-6:
            m.cells.extend(_rest_cells(end - cursor))
        measures.append(m)

    return measures


def _rest_cells(ql: float) -> list[Cell]:
    cells: list[Cell] = []
    remaining = ql
    while remaining > 1e-6:
        take = min(remaining, 4.0)
        ul, dotted, dashes = _duration_shape(take)
        cells.append(Cell(text="0", underlines=ul, dotted=dotted))
        cells.extend(Cell(text="-", is_dash=True) for _ in range(dashes))
        remaining -= take
    return cells


def _cell_svg(cell: Cell, x: float, y: float) -> str:
    """Draw one cell; x is its left edge, y the digit baseline."""
    cx = x + SLOT_W / 2
    parts = []

    if cell.is_dash:
        parts.append(
            f'<line x1="{cx-11:.1f}" y1="{y-8:.1f}" x2="{cx+11:.1f}" y2="{y-8:.1f}" '
            f'stroke="currentColor" stroke-width="2"/>'
        )
        return "".join(parts)

    if cell.accidental:
        sym = "♯" if cell.accidental == "#" else "♭"
        parts.append(
            f'<text x="{cx-13:.1f}" y="{y-9:.1f}" font-size="15" '
            f'text-anchor="middle">{sym}</text>'
        )

    parts.append(
        f'<text x="{cx:.1f}" y="{y:.1f}" font-size="{BASE_FONT}" '
        f'text-anchor="middle" font-family="Georgia, serif">{cell.text}</text>'
    )

    # Octave dots: above for higher, below for lower, clear of any underlines.
    if cell.octave > 0:
        for i in range(min(cell.octave, 3)):
            parts.append(f'<circle cx="{cx:.1f}" cy="{y-BASE_FONT-3-i*7:.1f}" r="2.2"/>')
    elif cell.octave < 0:
        below = y + 8 + cell.underlines * 5
        for i in range(min(-cell.octave, 3)):
            parts.append(f'<circle cx="{cx:.1f}" cy="{below+i*7:.1f}" r="2.2"/>')

    for i in range(cell.underlines):
        uy = y + 5 + i * 5
        parts.append(
            f'<line x1="{cx-12:.1f}" y1="{uy:.1f}" x2="{cx+12:.1f}" y2="{uy:.1f}" '
            f'stroke="currentColor" stroke-width="1.6"/>'
        )

    if cell.dotted:
        parts.append(f'<circle cx="{cx+15:.1f}" cy="{y-7:.1f}" r="2.2"/>')

    return "".join(parts)


def render_svg(measures: list[Measure], analysis: Analysis, title: str,
               subtitle: str = "") -> list[str]:
    """Lay measures into lines and pages. Returns one SVG string per page."""
    tonic_pc, tonic_name = _tonic_pc(analysis)
    usable = PAGE_W - 2 * MARGIN_X

    # Pack measures into lines that fit the page width.
    lines: list[list[Measure]] = []
    current: list[Measure] = []
    width = 0
    for m in measures:
        w = (m.width_slots + 1) * SLOT_W          # +1 for the barline gap
        if current and width + w > usable:
            lines.append(current)
            current, width = [], 0
        current.append(m)
        width += w
    if current:
        lines.append(current)

    header_h = 108
    lines_per_page = max(1, (PAGE_H - MARGIN_TOP - header_h - 40) // LINE_H)

    pages: list[str] = []
    for page_idx in range(0, len(lines), lines_per_page):
        chunk = lines[page_idx: page_idx + lines_per_page]
        body: list[str] = []
        y = MARGIN_TOP

        if page_idx == 0:
            body.append(
                f'<text x="{PAGE_W/2}" y="{y}" font-size="24" text-anchor="middle" '
                f'font-weight="600">{html.escape(title)}</text>'
            )
            y += 26
            if subtitle:
                body.append(
                    f'<text x="{PAGE_W/2}" y="{y}" font-size="13" '
                    f'text-anchor="middle" opacity="0.65">{html.escape(subtitle)}</text>'
                )
                y += 22
            body.append(
                f'<text x="{MARGIN_X}" y="{y}" font-size="15">'
                f'1 = {tonic_name}&#160;&#160;&#160;{analysis.beats_per_bar}/4'
                f'&#160;&#160;&#160;♩ = {round(analysis.tempo)}</text>'
            )
            y += 40
        else:
            y += 10

        for line in chunk:
            x = MARGIN_X
            baseline = y + 26
            body.append(
                f'<line x1="{x-6:.1f}" y1="{baseline-24:.1f}" x2="{x-6:.1f}" '
                f'y2="{baseline+8:.1f}" stroke="currentColor" stroke-width="1.4"/>'
            )
            for m in line:
                for cell in m.cells:
                    body.append(_cell_svg(cell, x, baseline))
                    x += SLOT_W
                x += SLOT_W * 0.4
                body.append(
                    f'<line x1="{x:.1f}" y1="{baseline-24:.1f}" x2="{x:.1f}" '
                    f'y2="{baseline+8:.1f}" stroke="currentColor" stroke-width="1.4"/>'
                )
                x += SLOT_W * 0.6
            y += LINE_H

        pages.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" '
            f'height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}" '
            f'style="color:#111"><rect width="100%" height="100%" fill="#fff"/>'
            f'<g fill="currentColor">{"".join(body)}</g></svg>'
        )

    return pages


def write_jianpu(notes: list[GridNote], analysis: Analysis, title: str,
                 outdir: Path, subtitle: str = "",
                 basename: str = "jianpu") -> dict:
    """Write per-page SVGs plus a printable HTML wrapper."""
    measures = melody_to_measures(notes, analysis)
    if not measures:
        return {}

    outdir.mkdir(parents=True, exist_ok=True)
    pages = render_svg(measures, analysis, title, subtitle)

    svg_paths = []
    for i, svg in enumerate(pages, 1):
        p = outdir / f"{basename}-{i}.svg"
        p.write_text(svg, encoding="utf-8")
        svg_paths.append(p)

    # A print-to-PDF wrapper: browsers give proper A4 pagination for free.
    html_doc = (
        '<!doctype html><meta charset="utf-8">'
        f'<title>{html.escape(title)} 簡譜</title>'
        '<style>@page{size:A4;margin:0}'
        'body{margin:0;background:#f4f4f5}'
        'svg{display:block;margin:0 auto 12px;background:#fff;'
        'box-shadow:0 1px 6px rgba(0,0,0,.15)}'
        '@media print{body{background:#fff}svg{box-shadow:none;margin:0;'
        'page-break-after:always}}</style>'
        + "".join(pages)
    )
    html_path = outdir / f"{basename}.html"
    html_path.write_text(html_doc, encoding="utf-8")

    return {"svg_pages": svg_paths, "html": html_path, "page_count": len(pages)}
