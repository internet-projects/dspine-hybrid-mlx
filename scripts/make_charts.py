"""Write the README / model-card charts as SVG (stdlib only). PNGs are rendered from these SVGs.

    python scripts/make_charts.py OUT_DIR
"""

from __future__ import annotations

import os
import sys

THEMES = {
    "light": dict(bg="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e4e3df",
                  series=["#b5b4ae", "#2a78d6", "#eb6834"]),
    "dark": dict(bg="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#34332f",
                 series=["#5f5e59", "#3987e5", "#d95926"]),
}
FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# rows: (label, [plain, dspine, ...], [speedup per DSpine bar from the unrounded measurements])
SPEED = {
    "title": "Decode speed, thinking on",
    "sub": "Qwen3.8-27B, greedy, 512 new tokens, MacBook Pro M5 Max",
    "series": ["Plain mlx-lm", "With DSpine"],
    "rows": [("4-bit", [31.9, 64.7], [2.03]), ("8-bit", [17.9, 52.1], [2.91]), ("bf16", [9.1, 39.1], [4.31])],
}
THINK_OFF = {
    "title": "Decode speed, thinking off",
    "sub": "Qwen3.8-27B, greedy, 512 new tokens, MacBook Pro M5 Max",
    "series": ["Plain mlx-lm", "DSpine (default)", "DSpine --adaptive"],
    "rows": [("4-bit", [32.0, 68.8, 71.0], [2.15, 2.22]), ("8-bit", [18.0, 52.8, 54.0], [2.93, 3.00])],
}
LONG = {
    "title": "Long prompts",
    "sub": "Thinking off, 256 new tokens after a 9k or 33k-token prompt",
    "series": ["Plain mlx-lm", "With DSpine"],
    "rows": [("4-bit, 9k", [30.6, 63.6], [2.08]), ("4-bit, 33k", [28.3, 57.3], [2.02]),
             ("8-bit, 9k", [17.5, 50.7], [2.89]), ("8-bit, 33k", [16.7, 47.7], [2.85])],
}


def bar(x0, y, w, h, r, fill):
    """Horizontal bar, square at the baseline, 4px rounded at the value end."""
    r = min(r, w / 2, h / 2)
    x1 = x0 + w
    return (f'<path d="M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + h - r} Q{x1},{y + h} {x1 - r},{y + h} '
            f'H{x0} Z" fill="{fill}"/>')


def chart(spec, theme, width=880):
    c = THEMES[theme]
    rows, names = spec["rows"], spec["series"]
    n = len(names)
    left, right, top = 132, 150, 112
    bh, gap, group_gap = 22, 2, 26
    gh = n * bh + (n - 1) * gap
    height = top + len(rows) * gh + (len(rows) - 1) * group_gap + 54
    vmax = max(max(r[1]) for r in rows)
    step = 10 if vmax <= 80 else 20
    axis_max = (int(vmax // step) + 1) * step
    plot_w = width - left - right
    sx = plot_w / axis_max
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
         f'font-family="{FONT}">',
         f'<rect width="{width}" height="{height}" fill="{c["bg"]}"/>',
         f'<text x="32" y="44" font-size="22" font-weight="600" fill="{c["text"]}">{spec["title"]}</text>',
         f'<text x="32" y="70" font-size="14" fill="{c["text2"]}">{spec["sub"]}</text>']
    lx = 32
    for k, label in enumerate(names):
        o.append(f'<rect x="{lx}" y="86" width="12" height="12" rx="2" fill="{c["series"][k]}"/>')
        o.append(f'<text x="{lx + 18}" y="97" font-size="13" fill="{c["text2"]}">{label}</text>')
        lx += 30 + 7.6 * len(label)
    plot_bottom = height - 40
    for t in range(0, axis_max + 1, step):
        x = left + t * sx
        o.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{plot_bottom}" stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{plot_bottom + 18}" font-size="12" fill="{c["text2"]}" text-anchor="middle">{t}</text>')
    o.append(f'<text x="{left + plot_w / 2:.1f}" y="{plot_bottom + 34}" font-size="12" fill="{c["text2"]}" '
             f'text-anchor="middle">tokens per second</text>')
    y = top
    for name, vals, speedups in rows:
        o.append(f'<text x="{left - 14}" y="{y + gh / 2 + 5:.1f}" font-size="15" font-weight="500" fill="{c["text"]}" '
                 f'text-anchor="end">{name}</text>')
        for k, v in enumerate(vals):
            yy = y + k * (bh + gap)
            o.append(bar(left, yy, v * sx, bh, 4, c["series"][k]))
            if k == 0:
                o.append(f'<text x="{left + v * sx + 8:.1f}" y="{yy + 16}" font-size="13" fill="{c["text2"]}">{v:.1f}</text>')
            else:
                o.append(f'<text x="{left + v * sx + 8:.1f}" y="{yy + 16}" font-size="13" fill="{c["text"]}">{v:.1f}'
                         f'<tspan font-weight="700" dx="8">×{speedups[k - 1]:.2f}</tspan></text>')
        y += gh + group_gap
    o.append("</svg>")
    return "\n".join(o)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    files = []
    for key, spec in (("speed", SPEED), ("thinking-off", THINK_OFF), ("long-context", LONG)):
        for theme in THEMES:
            svg = chart(spec, theme)
            path = os.path.join(out, f"{key}-{theme}.svg")
            open(path, "w").write(svg)
            files.append(path)
    print("\n".join(files))


if __name__ == "__main__":
    main()
