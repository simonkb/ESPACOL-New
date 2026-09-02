#!/usr/bin/env python3
"""Generate the editable and publication-rendered MOSAIC architecture figure.

The visual language deliberately mirrors ``paper/Optic-c.drawio`` while the
content is derived from the implemented MOSAIC forward pass.  The script emits
an editable draw.io document and a standalone SVG with the reference fundus
embedded, so neither artifact depends on external image paths.
"""

from __future__ import annotations

import argparse
import base64
from html import escape
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


CANVAS_W = 1600
CANVAS_H = 900

BLUE = "#6583B5"
TOKEN_BLUE = "#2570BD"
LIGHT_BLUE = "#CCE5FF"
MID_BLUE = "#99CCFF"
OUTLINE_BLUE = "#3399FF"
PALE_YELLOW = "#FFF2CC"
ORANGE = "#FF8000"
PEACH = "#FFCC99"
CORAL = "#FFCAC6"
CORAL_STROKE = "#9C4241"
GRAY = "#777777"
LIGHT_GRAY = "#F4F6F8"


def _image_uri(path: Path, *, canonical: bool) -> str:
    """Return a compact embedded image without requiring Pillow.

    macOS ``sips`` is used when available to keep the editable draw.io file
    reasonably small.  On other systems the original image is embedded and
    the SVG/draw.io viewport performs the display crop.  ``canonical`` is
    retained to make the two semantic uses explicit in the calling code.
    """
    del canonical
    payload = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    sips = shutil.which("sips")
    if sips:
        with tempfile.TemporaryDirectory(prefix="mosaic_arch_") as tmp_dir:
            compact = Path(tmp_dir) / "fundus.jpg"
            subprocess.run(
                [
                    sips,
                    "--resampleHeightWidthMax",
                    "640",
                    "--setProperty",
                    "format",
                    "jpeg",
                    str(path),
                    "--out",
                    str(compact),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            payload = compact.read_bytes()
            mime = "image/jpeg"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class Svg:
    def __init__(self, raw_uri: str, canonical_uri: str) -> None:
        self.raw_uri = raw_uri
        self.canonical_uri = canonical_uri
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{CANVAS_W}" height="{CANVAS_H}" '
            f'viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
            """
<defs>
  <marker id="arrow-black" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,4 L0,8 z" fill="#111111"/>
  </marker>
  <marker id="arrow-orange" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,4 L0,8 z" fill="#FF8000"/>
  </marker>
  <marker id="arrow-gray" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,4 L0,8 z" fill="#777777"/>
  </marker>
  <clipPath id="raw-clip"><rect x="30" y="61" width="135" height="135" rx="1"/></clipPath>
  <clipPath id="canonical-clip"><rect x="205" y="61" width="135" height="135" rx="1"/></clipPath>
  <clipPath id="proof-clip"><rect x="1315" y="294" width="220" height="190" rx="1"/></clipPath>
</defs>
<rect x="0" y="0" width="1600" height="900" fill="#FFFFFF"/>
""",
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "#FFFFFF",
        stroke: str = "#111111",
        radius: float = 10,
        width: float = 1.2,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{radius}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        lines: list[str],
        *,
        size: float = 12,
        leading: float | None = None,
        anchor: str = "middle",
        bold_first: bool = False,
        color: str = "#111111",
        family: str = "Arial,Helvetica,sans-serif",
        italic: bool = False,
    ) -> None:
        leading = leading or size * 1.28
        style = "font-style:italic;" if italic else ""
        self.add(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'fill="{color}" font-family="{family}" font-size="{size}" '
            f'style="{style}">'
        )
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else leading
            weight = ' font-weight="700"' if bold_first and index == 0 else ""
            self.add(
                f'<tspan x="{x}" dy="{dy}"{weight}>{escape(line)}</tspan>'
            )
        self.add("</text>")

    def arrow(
        self,
        points: list[tuple[float, float]],
        *,
        color: str = "#111111",
        width: float = 2,
        dashed: bool = False,
        marker: bool = True,
    ) -> None:
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x} {y}"
            for index, (x, y) in enumerate(points)
        )
        marker_id = (
            "arrow-orange" if color == ORANGE else
            "arrow-gray" if color == GRAY else
            "arrow-black"
        )
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        marker_attr = f' marker-end="url(#{marker_id})"' if marker else ""
        self.add(
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="miter"{dash}{marker_attr}/>'
        )

    def image(self, uri: str, x: float, y: float, w: float, h: float, clip: str) -> None:
        self.add(
            f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip})" '
            f'href="{uri}" xlink:href="{uri}"/>'
        )

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>", ""])


def _svg_figure(raw_uri: str, canonical_uri: str) -> str:
    s = Svg(raw_uri, canonical_uri)

    # Input and canonicalization group.
    s.rect(15, 20, 340, 285, stroke="#A9C4EB", radius=24, width=1.2)
    s.text(185, 43, ["Full-Canvas Canonicalization"], size=14, bold_first=True)
    s.image(raw_uri, 30, 61, 135, 135, "raw-clip")
    s.image(canonical_uri, 205, 61, 135, 135, "canonical-clip")
    s.rect(30, 61, 135, 135, fill="none", radius=0, width=1)
    s.rect(205, 61, 135, 135, fill="none", radius=0, width=1)
    for offset in range(1, 9):
        x = 205 + 135 * offset / 9
        y = 61 + 135 * offset / 9
        s.arrow([(x, 61), (x, 196)], color="#FFFFFF", width=0.55, marker=False)
        s.arrow([(205, y), (340, y)], color="#FFFFFF", width=0.55, marker=False)
    s.add('<ellipse cx="272.5" cy="128.5" rx="64" ry="64" fill="none" stroke="#66B3FF" stroke-width="2"/>')
    s.arrow([(170, 128), (198, 128)], width=2)
    s.text(97, 216, ["Input fundus image", "X: N × 3 × H × W"], size=11)
    s.text(272, 216, ["Dominant-field crop", "+ canonical 896 × 896"], size=11)
    s.rect(60, 260, 250, 30, fill="#F5F8FC", stroke="#6CA6E8", radius=6)
    s.text(185, 280, ["Fixed retinal support mask Mpx  (not a lesion mask)"], size=10.5)

    # Encoder wedge and pointwise adapter.
    s.add(
        f'<polygon points="385,55 385,265 570,205 570,115" '
        f'fill="{BLUE}" stroke="{BLUE}" stroke-width="1.2"/>'
    )
    s.text(
        468,
        126,
        [
            "Spatially Bounded",
            "EfficientNetV2-S",
            "stem–stage 3 • pretrained",
            "stride 8 • RF 95 × 95",
            "FrozenBN • no SE/GAP",
        ],
        size=11.5,
        leading=17,
        bold_first=True,
        color="#FFFFFF",
    )
    s.arrow([(355, 150), (378, 150)], width=2.2)
    s.rect(392, 285, 172, 46, fill=MID_BLUE, stroke="#6CA6E8", radius=7)
    s.text(478, 305, ["Tap feature map", "Ftap: N × 64 × 112 × 112"], size=10.5, bold_first=True)
    s.arrow([(478, 265), (478, 279)], width=2)
    s.rect(392, 350, 172, 76, fill=LIGHT_BLUE, stroke="#6CA6E8", radius=8)
    s.text(
        478,
        371,
        ["Pointwise Residual MLP", "64 → 128 → 256 → 128", "no spatial mixing"],
        size=10.5,
        bold_first=True,
    )
    s.arrow([(478, 331), (478, 344)], width=2)
    s.rect(382, 449, 192, 63, fill=ORANGE, stroke=ORANGE, radius=7)
    s.text(
        478,
        470,
        ["Local evidence lattice H", "N × 12,544 × 128", "9,864 proof-valid cells"],
        size=10.5,
        bold_first=True,
    )
    s.arrow([(478, 426), (478, 443)], width=2)

    # Main MOSAIC module.
    s.rect(595, 20, 665, 785, stroke=OUTLINE_BLUE, radius=18, width=1.3)
    s.text(
        927,
        45,
        ["MOSAIC — Minimum Ordinal Sufficient Attribution by Intervention and Counting"],
        size=14,
        bold_first=True,
    )
    s.rect(620, 70, 190, 137, fill=LIGHT_BLUE, stroke="#111111", radius=12)
    s.text(
        715,
        94,
        [
            "Shared Local Ordinal State Head",
            "Linear 128 → 5 + softmax",
            "q(n,i,s) = P(Lᵢ = s)",
            "N × 12,544 × 5",
        ],
        size=11,
        leading=21,
        bold_first=True,
    )
    for index, color in enumerate(["#2D6FB7", "#4E8DCC", "#72A6D8", "#A0C5E8", "#D3E5F7"]):
        s.rect(658 + index * 24, 174, 18, 18, fill=color, stroke=color, radius=2)

    s.rect(840, 70, 205, 137, fill=LIGHT_BLUE, stroke="#111111", radius=12)
    s.text(
        942,
        94,
        [
            "Structurally Nested Witnesses",
            "λ[n,i,k] = Σ_(s>k) q[n,i,s]",
            "λ[i,0] ≥ λ[i,1] ≥ λ[i,2] ≥ λ[i,3]",
            "N × 12,544 × 4",
        ],
        size=11,
        leading=21,
        bold_first=True,
        family="Arial,Helvetica,sans-serif",
    )
    for row in range(4):
        for col in range(9):
            opacity = max(0.18, 0.92 - 0.08 * col - 0.09 * row)
            color = TOKEN_BLUE if col < 5 - row else "#DCE9F7"
            s.rect(862 + col * 17, 171 + row * 7, 12, 5, fill=color, stroke=color, radius=1, opacity=opacity)

    s.rect(1070, 70, 165, 137, fill="#EEF6FF", stroke="#6CA6E8", radius=12)
    s.text(1152, 94, ["Four Boundary", "Evidence Maps"], size=11, bold_first=True)
    for row in range(6):
        for col in range(8):
            base = 0.12 + ((row * 7 + col * 5) % 9) / 12
            color = ORANGE if (row, col) in {(1, 5), (2, 5), (3, 2), (4, 3)} else TOKEN_BLUE
            s.rect(1093 + col * 15, 142 + row * 10, 11, 7, fill=color, stroke=color, radius=1, opacity=base)
    s.arrow([(574, 480), (595, 480), (595, 138), (613, 138)], color=ORANGE, width=2.4)
    s.arrow([(810, 138), (833, 138)], width=2)
    s.arrow([(1045, 138), (1063, 138)], width=2)

    # Counting and dense target.
    s.rect(620, 265, 250, 174, fill=LIGHT_BLUE, stroke="#111111", radius=12)
    s.text(
        745,
        290,
        [
            "Exact Truncated Poisson–Binomial Counter",
            "Z[i,k] ~ Bernoulli(λ[i,k])",
            "C[n,k] = Σ_(i in V) Z[i,k]",
            "P(C=0), …, P(C=31), P(C≥32)",
            "exact block-tree recurrence • FP32",
        ],
        size=11,
        leading=24,
        bold_first=True,
    )
    s.rect(620, 462, 250, 78, fill=PEACH, stroke="#E9A45A", radius=10)
    s.text(
        745,
        484,
        [
            "Learnable Boundary Count Mixtures",
            "α[k] = softmax(η[k]) in simplex(32)   •   α: 4 × 32",
        ],
        size=11,
        leading=25,
        bold_first=True,
    )
    s.rect(900, 275, 145, 112, fill=MID_BLUE, stroke="#111111", radius=11)
    s.text(
        972,
        300,
        [
            "Dense Evidence",
            "cD[n,k] = Σ_r α[k,r]",
            "P(C[n,k] ≥ r)",
            "N × 4",
        ],
        size=10.5,
        leading=20,
        bold_first=True,
    )
    s.arrow([(942, 207), (942, 237), (745, 237), (745, 258)], width=2)
    s.arrow([(870, 352), (893, 352)], width=2)
    s.arrow([(745, 462), (745, 446)], width=2)

    # Minimum dual proof.
    s.rect(885, 410, 350, 180, fill="#FFF4E6", stroke=ORANGE, radius=13, width=2)
    s.text(
        1060,
        435,
        [
            "Deterministic Minimum Top-Prefix Proof",
            "sort λ[i,k] descending; select smallest prefix m*",
            "Sufficiency:  cR[k] ≥ cD[k] − ε",
            "Collective necessity:  cD[k] − cC[k] ≥ ρ(cD[k] − ε)",
            "ε = 0.02   •   ρ = 0.5",
            "M*: N × 12,544 × 4   •   m*: N × 4",
        ],
        size=10.5,
        leading=25,
        bold_first=True,
    )
    s.arrow([(1045, 331), (1060, 331), (1060, 403)], width=2)
    s.arrow([(1045, 138), (1058, 138), (1058, 403)], width=2)
    s.arrow([(870, 501), (878, 501)], width=2)

    # Proof-exclusive retained replay and ordinal cascade.
    s.rect(630, 625, 255, 100, fill=LIGHT_BLUE, stroke=ORANGE, radius=11, width=2)
    s.text(
        757,
        650,
        [
            "Retained-Proof Circuit Replay",
            "same PB counter on M* × λ",
            "c[k] = cR[k]   and stable log s[k]",
        ],
        size=11,
        leading=24,
        bold_first=True,
    )
    s.rect(930, 615, 290, 130, fill=PALE_YELLOW, stroke="#111111", radius=12)
    s.text(
        1075,
        640,
        [
            "Proof-Only Ordinal Cascade",
            "P(Y>k) = product_(j=0..k) c[j]",
            "P(Y=0), …, P(Y=4)",
            "E[Y] = sum_(k=0..3) P(Y>k)",
            "ŷ = round(E[Y])   (safe default)",
            "6-rule comparison: diagnostic audit only",
        ],
        size=9.5,
        leading=19,
        bold_first=True,
    )
    s.arrow([(1060, 590), (1060, 607), (757, 607), (757, 618)], color=ORANGE, width=2.6)
    s.arrow([(885, 675), (923, 675)], color=ORANGE, width=2.6)

    # Training objective.
    s.rect(1285, 25, 300, 180, fill=CORAL, stroke=CORAL_STROKE, radius=18, width=1.5)
    s.text(
        1435,
        55,
        [
            "Balanced At-Risk Continuation NLL",
            "Lproj = −Σ_(k<y) w[k,1] log c[k]",
            "        − 1[y<4] w[y,0] log s[y]",
            "L = Lproj + 0.1 Ldense",
            "effective-number transition weights",
        ],
        size=11.5,
        leading=27,
        bold_first=True,
        family="Arial,Helvetica,sans-serif",
    )
    s.arrow([(885, 660), (910, 660), (910, 600), (1265, 600), (1265, 125), (1278, 125)], color=CORAL_STROKE, width=1.7, dashed=True)
    s.arrow([(972, 275), (972, 230), (1265, 230), (1265, 170), (1278, 170)], color=GRAY, width=1.5, dashed=True)
    s.text(1185, 224, ["dense auxiliary • training only"], size=9.5, anchor="middle", color=GRAY, italic=True)

    # Replayable certificate with schematic fine-grid proof.
    s.rect(1285, 225, 300, 405, fill="#FFFFFF", stroke="#111111", radius=12, width=1.2)
    s.text(1435, 253, ["Minimum Ordinal Proof Certificate (inference)"], size=13, bold_first=True)
    s.image(canonical_uri, 1315, 294, 220, 190, "proof-clip")
    s.rect(1315, 294, 220, 190, fill="none", radius=0, width=1)
    rows, cols = 10, 12
    cw, ch = 220 / cols, 190 / rows
    for col in range(1, cols):
        x = 1315 + col * cw
        s.arrow([(x, 294), (x, 484)], color="#FFFFFF", width=0.5, marker=False)
    for row in range(1, rows):
        y = 294 + row * ch
        s.arrow([(1315, y), (1535, y)], color="#FFFFFF", width=0.5, marker=False)
    selected = {(2, 8), (3, 8), (4, 7), (5, 4), (6, 4), (7, 5), (7, 6)}
    for row, col in selected:
        s.rect(
            1315 + col * cw + 1,
            294 + row * ch + 1,
            cw - 2,
            ch - 2,
            fill=ORANGE,
            stroke=ORANGE,
            radius=1,
            opacity=0.58,
        )
    s.text(1425, 500, ["schematic selected witnesses M*[k]"], size=9.5, italic=True, color=GRAY)
    s.text(
        1435,
        527,
        [
            "Per boundary: selected indices • proof size m*[k]",
            "dense / retained / complement scores: cD, cR, cC",
            "sufficiency gap • complement drop",
            "optional exact fixed-proof intervention Δ[i,k]",
            "integrity hash + independent numerical replay",
        ],
        size=9.8,
        leading=20,
    )
    s.arrow([(1235, 500), (1278, 500)], color=ORANGE, width=2.4)

    # Prediction outputs.
    s.rect(1285, 650, 300, 165, fill="#FFFFFF", stroke="#111111", radius=12, width=1.2)
    s.text(1435, 678, ["Ordinal Outputs (inference)"], size=13, bold_first=True)
    s.rect(1310, 700, 250, 43, fill=LIGHT_BLUE, stroke="#6CA6E8", radius=7)
    s.text(1435, 718, ["c[0..3]  •  P(Y>k)  •  P(Y=0..4)"], size=10.5)
    s.rect(1310, 755, 250, 43, fill=ORANGE, stroke=ORANGE, radius=7)
    s.text(1435, 774, ["round(E[Y])  →  predicted grade ŷ"], size=11, bold_first=True)
    s.arrow([(1220, 680), (1278, 680)], color=ORANGE, width=2.6)

    # Footer claim boundary.
    s.rect(260, 838, 1080, 38, fill=LIGHT_GRAY, stroke="#C5CCD3", radius=8)
    s.text(
        800,
        861,
        [
            "Fine-grid local witnesses: stride-8 centers with 95×95 receptive-field support — not pixel segmentation   •   Prediction has no global classifier bypass",
        ],
        size=10.5,
        color="#374151",
    )
    return s.finish()


class Drawio:
    def __init__(self, raw_uri: str, canonical_uri: str) -> None:
        self.raw_uri = raw_uri
        self.canonical_uri = canonical_uri
        self.mxfile = ET.Element("mxfile", {"host": "Electron"})
        self.diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {"name": "MOSAIC Architecture", "id": "mosaic-architecture-v1"},
        )
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1883",
                "dy": "1818",
                "grid": "0",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(CANVAS_W),
                "pageHeight": str(CANVAS_H),
                "background": "#FFFFFF",
                "math": "1",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.counter = 1

    def _id(self, prefix: str = "m") -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def vertex(self, value: str, x: float, y: float, w: float, h: float, style: str, *, ident: str | None = None) -> str:
        ident = ident or self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": ident, "value": value, "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return ident

    def edge(self, source: str, target: str, *, color: str = "#111111", width: float = 2, dashed: bool = False) -> str:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            "jettySize=auto;html=1;endArrow=classic;endFill=1;"
            f"strokeColor={color};strokeWidth={width};"
            f"dashed={1 if dashed else 0};"
        )
        ident = self._id("e")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": ident, "style": style, "edge": "1", "parent": "1", "source": source, "target": target},
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        return ident

    def finish(self) -> str:
        ET.indent(self.mxfile, space="  ")
        return ET.tostring(self.mxfile, encoding="unicode") + "\n"


def _html(title: str, *lines: str, size: int = 11) -> str:
    body = "<br>".join(lines)
    return f'<div style="text-align:center"><b>{title}</b><br><font style="font-size:{size}px">{body}</font></div>'


def _drawio_figure(raw_uri: str, canonical_uri: str) -> str:
    d = Drawio(raw_uri, canonical_uri)
    # mxGraph stores styles as semicolon-delimited strings, so the standard
    # ``;base64`` data-URI marker would terminate the image style early.
    # diagrams.net's native embedded-image convention omits that marker.
    raw_drawio_uri = raw_uri.replace(";base64,", ",", 1)
    canonical_drawio_uri = canonical_uri.replace(";base64,", ",", 1)
    group_style = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#3399FF;strokeWidth=1.2;"
    box = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#CCE5FF;strokeColor=#111111;fontSize=12;align=center;verticalAlign=middle;"
    peach = box.replace("#CCE5FF", PEACH).replace("#111111", "#E9A45A")
    yellow = box.replace("#CCE5FF", PALE_YELLOW)
    orange_box = box.replace("#CCE5FF", "#FFF4E6").replace("strokeWidth=", "strokeWidth=2;").replace("#111111", ORANGE)
    loss = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#FFCAC6;strokeColor=#9C4241;strokeWidth=1.5;fontSize=12;align=center;verticalAlign=middle;"
    output = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#111111;strokeWidth=1.2;fontSize=12;align=center;verticalAlign=middle;"
    text_only = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;"

    d.vertex("", 15, 20, 340, 285, "rounded=1;arcSize=8;fillColor=none;strokeColor=#A9C4EB;strokeWidth=1.2;", ident="input_group")
    d.vertex("<b>Full-Canvas Canonicalization</b>", 75, 25, 220, 25, text_only)
    raw = d.vertex("", 30, 61, 135, 135, f"shape=image;imageAspect=0;aspect=fixed;image={raw_drawio_uri};strokeColor=#111111;", ident="raw_image")
    canonical = d.vertex("", 205, 61, 135, 135, f"shape=image;imageAspect=0;aspect=fixed;image={canonical_drawio_uri};strokeColor=#111111;", ident="canonical_image")
    input_grid_style = "shape=rectangle;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FFFFFF;opacity=70;"
    for offset in range(1, 9):
        d.vertex("", 205 + offset * 15, 61, 0.5, 135, input_grid_style)
        d.vertex("", 205, 61 + offset * 15, 135, 0.5, input_grid_style)
    d.vertex("", 211, 65, 123, 123, "ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#66B3FF;strokeWidth=2;")
    d.vertex("Input fundus image<br>X: N × 3 × H × W", 30, 202, 135, 42, "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;")
    d.vertex("Dominant-field crop<br>+ canonical 896 × 896", 200, 202, 145, 42, "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;")
    d.vertex("Fixed retinal support mask Mpx&nbsp; (not a lesion mask)", 60, 260, 250, 30, "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5F8FC;strokeColor=#6CA6E8;fontSize=10;")
    d.edge(raw, canonical)

    encoder = d.vertex(
        '<div style="color:#FFFFFF;text-align:center"><b>Spatially Bounded<br>EfficientNetV2-S</b><br><font style="font-size:11px">stem–stage 3 • pretrained<br>stride 8 • RF 95 × 95<br>FrozenBN • no SE/GAP</font></div>',
        385, 55, 185, 210,
        "shape=trapezoid;direction=east;whiteSpace=wrap;html=1;fillColor=#6583B5;strokeColor=#6583B5;fontSize=12;",
        ident="encoder",
    )
    d.edge(canonical, encoder)
    tap = d.vertex(_html("Tap feature map", "Ftap: N × 64 × 112 × 112"), 392, 285, 172, 46, box.replace("#CCE5FF", MID_BLUE), ident="tap")
    mlp = d.vertex(_html("Pointwise Residual MLP", "64 → 128 → 256 → 128", "no spatial mixing"), 392, 350, 172, 76, box, ident="mlp")
    lattice = d.vertex(_html("Local evidence lattice H", "N × 12,544 × 128", "9,864 proof-valid cells"), 382, 449, 192, 63, box.replace("#CCE5FF", ORANGE).replace("#111111", ORANGE), ident="lattice")
    d.edge(encoder, tap)
    d.edge(tap, mlp)
    d.edge(mlp, lattice)

    d.vertex("", 595, 20, 665, 785, group_style, ident="mosaic_group")
    d.vertex("<b>MOSAIC — Minimum Ordinal Sufficient Attribution by Intervention and Counting</b>", 625, 27, 605, 28, text_only)
    state = d.vertex(_html("Shared Local Ordinal State Head", "Linear 128 → 5 + softmax", "$$q_{n,i,s}=P(L_{n,i}=s)$$", "N × 12,544 × 5"), 620, 70, 190, 137, box, ident="state")
    witness = d.vertex(_html("Structurally Nested Witnesses", "$$\\lambda_{n,i,k}=\\sum_{s>k}q_{n,i,s}$$", "λᵢ,₀ ≥ λᵢ,₁ ≥ λᵢ,₂ ≥ λᵢ,₃", "N × 12,544 × 4"), 840, 70, 205, 137, box, ident="witness")
    maps = d.vertex("", 1070, 70, 165, 137, box.replace("#CCE5FF", "#EEF6FF").replace("#111111", "#6CA6E8"), ident="maps")
    d.vertex("<b>Four Boundary<br>Evidence Maps</b>", 1080, 82, 145, 42, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=12;")
    map_cells = {(1, 5), (2, 5), (3, 2), (4, 3)}
    for row in range(6):
        for col in range(8):
            fill = ORANGE if (row, col) in map_cells else ("#2570BD" if col < 5 else "#DCE9F7")
            d.vertex("", 1093 + col * 15, 142 + row * 10, 11, 7, f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={fill};")
    d.edge(lattice, state, color=ORANGE, width=2.4)
    d.edge(state, witness)
    d.edge(witness, maps)

    counter = d.vertex(_html("Exact Truncated Poisson–Binomial Counter", "Zᵢ,ₖ ~ Bernoulli(λᵢ,ₖ)", "Cₙ,ₖ = Σᵢ∈V Zᵢ,ₖ", "P(C=0), …, P(C=31), P(C≥32)", "exact block-tree recurrence • FP32"), 620, 265, 250, 174, box, ident="counter")
    alpha = d.vertex(_html("Learnable Boundary Count Mixtures", "αₖ = softmax(ηₖ) ∈ Δ³¹", "α: 4 × 32"), 620, 462, 250, 78, peach, ident="alpha")
    dense = d.vertex(_html("Dense Evidence", "$$c^D_{n,k}=\\sum_r\\alpha_{k,r}P(C_{n,k}\\ge r)$$", "N × 4"), 900, 275, 145, 112, box.replace("#CCE5FF", MID_BLUE), ident="dense")
    d.edge(witness, counter)
    d.edge(alpha, counter)
    d.edge(counter, dense)

    proof = d.vertex(_html("Deterministic Minimum Top-Prefix Proof", "sort λᵢ,ₖ descending; select smallest prefix m*", "Sufficiency: cᴿₖ ≥ cᴰₖ − ε", "Collective necessity: cᴰₖ − cᶜₖ ≥ ρ(cᴰₖ − ε)", "ε=0.02 • ρ=0.5", "M*: N × 12,544 × 4 • m*: N × 4"), 885, 410, 350, 180, orange_box, ident="proof")
    d.edge(witness, proof)
    d.edge(dense, proof)
    d.edge(alpha, proof)

    retained = d.vertex(_html("Retained-Proof Circuit Replay", "same PB counter on M* ⊙ λ", "cₖ=cᴿₖ and stable log sₖ"), 630, 625, 255, 100, box.replace("strokeColor=#111111", f"strokeColor={ORANGE};strokeWidth=2"), ident="retained")
    cascade = d.vertex(_html("Proof-Only Ordinal Cascade", "P(Y&gt;k) = product(j=0..k) c[j]", "P(Y=0), …, P(Y=4)", "E[Y] = sum(k=0..3) P(Y&gt;k)", "ŷ = round(E[Y]) (safe default)", "6-rule comparison: diagnostic audit only"), 930, 615, 290, 130, yellow, ident="cascade")
    d.edge(proof, retained, color=ORANGE, width=2.6)
    d.edge(retained, cascade, color=ORANGE, width=2.6)

    loss_box = d.vertex(_html("Balanced At-Risk Continuation NLL", "Lproj = −Σₖ<ᵧ wₖ,₁ log cₖ", "− 1[y<4] wᵧ,₀ log sᵧ", "L = Lproj + 0.1 Ldense", "effective-number transition weights"), 1285, 25, 300, 180, loss, ident="loss")
    d.edge(retained, loss_box, color=CORAL_STROKE, width=1.7, dashed=True)
    d.edge(dense, loss_box, color=GRAY, width=1.5, dashed=True)

    certificate = d.vertex("", 1285, 225, 300, 405, output, ident="certificate")
    d.vertex("<b>Minimum Ordinal Proof Certificate (inference)</b>", 1298, 240, 274, 28, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=13;")
    d.vertex("", 1315, 294, 220, 190, f"shape=image;imageAspect=0;aspect=fixed;image={canonical_drawio_uri};strokeColor=#111111;", ident="proof_image")
    grid_style = "shape=rectangle;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FFFFFF;opacity=75;"
    for col in range(1, 12):
        d.vertex("", 1315 + col * (220 / 12), 294, 0.6, 190, grid_style)
    for row in range(1, 10):
        d.vertex("", 1315, 294 + row * 19, 220, 0.6, grid_style)
    selected = {(2, 8), (3, 8), (4, 7), (5, 4), (6, 4), (7, 5), (7, 6)}
    selected_style = "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FFCC99;strokeColor=#FF8000;strokeWidth=2;opacity=85;"
    for row, col in selected:
        d.vertex("", 1315 + col * (220 / 12) + 1, 294 + row * 19 + 1, (220 / 12) - 2, 17, selected_style)
    d.vertex("<i>schematic selected witnesses M*[k]</i>", 1320, 488, 210, 20, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=9;fontColor=#777777;")
    d.vertex("Per boundary: selected indices • proof size m*[k]<br>dense / retained / complement: cD, cR, cC<br>sufficiency gap • complement drop<br>optional fixed-proof intervention Δ[i,k]<br>integrity hash + independent numerical replay", 1300, 520, 270, 92, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;")
    d.edge(proof, certificate, color=ORANGE, width=2.4)

    ordinal_output = d.vertex("", 1285, 650, 300, 165, output, ident="ordinal_output")
    d.vertex("<b>Ordinal Outputs (inference)</b>", 1315, 665, 240, 28, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=13;")
    d.vertex("c[0..3] • P(Y&gt;k) • P(Y=0..4)", 1310, 700, 250, 43, box)
    d.vertex("<b>round(E[Y]) → predicted grade ŷ</b>", 1310, 755, 250, 43, box.replace("#CCE5FF", ORANGE).replace("#111111", ORANGE))
    d.edge(cascade, ordinal_output, color=ORANGE, width=2.6)

    d.vertex("Fine-grid local witnesses: stride-8 centers with 95×95 receptive-field support — not pixel segmentation&nbsp;&nbsp; • &nbsp;&nbsp;Prediction has no global classifier bypass", 260, 838, 1080, 38, "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F4F6F8;strokeColor=#C5CCD3;fontSize=10;align=center;verticalAlign=middle;")
    return d.finish()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fundus",
        type=Path,
        default=root / "Datasets/aptos2019-blindness-detection/train_images/46923eea9a4e.png",
    )
    parser.add_argument("--output_dir", type=Path, default=root / "paper")
    args = parser.parse_args()
    if not args.fundus.is_file():
        raise FileNotFoundError(f"fundus image not found: {args.fundus}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_uri = _image_uri(args.fundus, canonical=False)
    canonical_uri = _image_uri(args.fundus, canonical=True)
    svg_path = args.output_dir / "MOSAIC_architecture.svg"
    drawio_path = args.output_dir / "MOSAIC_architecture.drawio"
    svg_path.write_text(_svg_figure(raw_uri, canonical_uri), encoding="utf-8")
    drawio_path.write_text(_drawio_figure(raw_uri, canonical_uri), encoding="utf-8")
    print(svg_path)
    print(drawio_path)


if __name__ == "__main__":
    main()
