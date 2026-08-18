#!/usr/bin/env python3
"""Render the cross-domain LoRA/full TTT comparison as a dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


TASKS = [
    ("erdos", "Erdős", -1),
    ("ac1", "AC1", -1),
    ("ac2", "AC2", 1),
    ("circle26", "Circle 26", 1),
    ("circle32", "Circle 32", 1),
    ("trimul", "TriMul", -1),
    ("mla", "MLA", -1),
    ("ahc039", "AHC039", 1),
    ("ahc058", "AHC058", 1),
    ("denoising", "Denoising", -1),
]
CURVE_TASKS = {"erdos", "ac1", "ac2", "circle26", "circle32", "ahc058"}

INK = "#172033"
MUTED = "#657086"
GRID = "#d8dee9"
FULL = "#2563eb"
LORA = "#e4572e"
POS = "#0f9d78"
NEG = "#d64550"
TIE = "#8a94a6"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, *, size: int = 14, anchor: str = "start", weight: int = 400, fill: str = INK) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = GRID, width: float = 1, dash: str | None = None) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{extra}/>'


def objective_delta(full: float, lora: float, direction: int) -> float:
    return direction * (lora - full) / max(abs(full), 1e-12) * 100


def progress(curve: list[float], direction: int) -> list[float]:
    if not curve:
        return []
    base = curve[0]
    return [direction * (value - base) / max(abs(base), 1e-12) * 100 for value in curve]


def horizontal_bars(parts: list[str], rows: list[tuple[str, float]], x: float, y: float, width: float, title_value: str, unit: str) -> None:
    parts.append(text(x, y, title_value, size=18, weight=700))
    chart_top = y + 30
    row_h = 30
    max_abs = max([abs(value) for _, value in rows] + [1.0]) * 1.12
    zero_x = x + width / 2
    parts.append(line(zero_x, chart_top - 8, zero_x, chart_top + row_h * len(rows) - 5, stroke=MUTED, width=1.2))
    for i, (label, value) in enumerate(rows):
        cy = chart_top + i * row_h + 8
        parts.append(text(x + 88, cy + 4, label, size=12, anchor="end", fill=MUTED))
        bar_max = width / 2 - 110
        bar_w = abs(value) / max_abs * bar_max
        bx = zero_x if value >= 0 else zero_x - bar_w
        color = POS if value > 0.005 else NEG if value < -0.005 else TIE
        parts.append(f'<rect x="{bx:.1f}" y="{cy - 8:.1f}" width="{max(bar_w, 1):.1f}" height="16" rx="2" fill="{color}"/>')
        label_x = zero_x + (bar_w + 6 if value >= 0 else -bar_w - 6)
        anchor = "start" if value >= 0 else "end"
        rendered = f"{value:+.2f}{unit}" if abs(value) >= 0.005 else "tie"
        parts.append(text(label_x, cy + 4, rendered, size=11, anchor=anchor, weight=600, fill=color))
    footer_y = chart_top + row_h * len(rows) + 8
    parts.append(text(zero_x - 7, footer_y, "Full better", size=11, anchor="end", fill=MUTED))
    parts.append(text(zero_x + 7, footer_y, "LoRA better", size=11, fill=MUTED))


def curve_panel(parts: list[str], label: str, full: list[float], lora: list[float], direction: int, x: float, y: float, width: float, height: float) -> None:
    parts.append(text(x, y, label, size=14, weight=700))
    px, py, pw, ph = x + 36, y + 16, width - 46, height - 38
    full_p, lora_p = progress(full, direction), progress(lora, direction)
    all_values = full_p + lora_p + [0.0]
    low, high = min(all_values), max(all_values)
    pad = max((high - low) * 0.12, 0.1)
    low, high = low - pad, high + pad
    parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="#ffffff" stroke="{GRID}"/>')
    zero_y = py + (high / (high - low)) * ph
    if py <= zero_y <= py + ph:
        parts.append(line(px, zero_y, px + pw, zero_y, stroke=GRID, dash="3 3"))
    parts.append(text(px - 5, py + 8, f"{high:.1f}%", size=10, anchor="end", fill=MUTED))
    parts.append(text(px - 5, py + ph, f"{low:.1f}%", size=10, anchor="end", fill=MUTED))

    def draw(values: list[float], color: str) -> None:
        if not values:
            return
        points = []
        for i, value in enumerate(values):
            xx = px + (i / max(len(values) - 1, 1)) * pw
            yy = py + (high - value) / (high - low) * ph
            points.append((xx, yy))
        joined = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        parts.append(f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>')
        for xx, yy in points:
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.8" fill="{color}"/>')

    draw(full_p, FULL)
    draw(lora_p, LORA)
    parts.append(text(px, py + ph + 16, "step 0", size=10, fill=MUTED))
    parts.append(text(px + pw, py + ph + 16, f"step {max(len(full_p), len(lora_p)) - 1}", size=10, anchor="end", fill=MUTED))


def render(data: dict) -> str:
    complete = []
    objective_rows = []
    validity_rows = []
    for key, label, direction in TASKS:
        pair = data.get(key, {})
        full, lora = pair.get("full", {}), pair.get("lora", {})
        if full.get("final_best") is None or lora.get("final_best") is None:
            continue
        complete.append((key, label, direction, full, lora))
        objective_rows.append((label, objective_delta(full["final_best"], lora["final_best"], direction)))
        full_rate, lora_rate = full.get("valid_rate"), lora.get("valid_rate")
        if full_rate is not None and lora_rate is not None:
            validity_rows.append((label, (lora_rate - full_rate) * 100))

    curves = [item for item in complete if item[0] in CURVE_TASKS]
    width = 1280
    curve_rows = (len(curves) + 2) // 3
    height = 520 + curve_rows * 220
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Qwen3-4B TTT full-parameter versus rank-32 LoRA</title>',
        '<desc id="desc">Cross-domain final objective, validity rate, and normalized best-so-far progress comparison.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif">',
        text(50, 52, "Qwen3-4B TTT: full parameter vs LoRA r=32", size=28, weight=750),
        text(50, 80, "Matched online-search budgets; positive bars favor LoRA. One run per method; trajectories are regenerated online.", size=14, fill=MUTED),
    ]
    horizontal_bars(parts, objective_rows, 50, 125, 565, "Final objective advantage", "%")
    horizontal_bars(parts, validity_rows, 665, 125, 565, "Valid-rollout rate difference", " pp")
    curves_y = 500
    parts.append(text(50, curves_y - 28, "Best-so-far progress from each run's step 0", size=18, weight=700))
    parts.append(line(875, curves_y - 34, 905, curves_y - 34, stroke=FULL, width=3))
    parts.append(text(913, curves_y - 29, "Full", size=12, fill=MUTED))
    parts.append(line(975, curves_y - 34, 1005, curves_y - 34, stroke=LORA, width=3))
    parts.append(text(1013, curves_y - 29, "LoRA r=32", size=12, fill=MUTED))
    panel_w, panel_h = 386, 185
    for i, (_, label, direction, full, lora) in enumerate(curves):
        col, row = i % 3, i // 3
        curve_panel(parts, label, full.get("best_curve", []), lora.get("best_curve", []), direction, 50 + col * 410, curves_y + row * 220, panel_w, panel_h)
    parts.append(text(50, height - 24, "Objective bars compare absolute final quality; curves compare within-run improvement and do not imply identical sampled trajectories.", size=12, fill=MUTED))
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("docs/ttt_lora_vs_full_results.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/ttt-lora-vs-full.svg"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data))
    print(args.output)


if __name__ == "__main__":
    main()
