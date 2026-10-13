"""Render :class:`~gazegrounded.data.ui_specs.UISpec` objects to PNG screenshots with Pillow.

The renderings are deliberately simple wireframe-style interfaces. They are good enough for
(a) a reproducible demo and (b) controlled study stimuli where only one design factor changes
between the clean and friction variants.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gazegrounded.data.ui_specs import Element, UISpec

_CARD_BORDER = "#D5DAE1"
_TEXT = "#1F2328"


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _px(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)


def _text(draw: ImageDraw.ImageDraw, xy, text, size, fill, bold=False, anchor="la", spacing=6):
    kwargs = {"stroke_width": 1, "stroke_fill": fill} if bold else {}
    draw.multiline_text(
        xy, text, font=_font(size), fill=fill, anchor=anchor, spacing=spacing, **kwargs
    )


def _draw_icon(draw: ImageDraw.ImageDraw, box, glyph: str, color: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, outline="#C9D0CF", width=2, fill="#F4F6F6")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = min(x1 - x0, y1 - y0) * 0.28
    if glyph == "lock":
        draw.rectangle((cx - r, cy - r * 0.1, cx + r, cy + r), fill=color)
        draw.arc(
            (cx - r * 0.7, cy - r * 1.1, cx + r * 0.7, cy + r * 0.3), 180, 360, fill=color, width=3
        )
    elif glyph == "pin":
        for i in range(3):
            for j in range(3):
                px, py = cx + (i - 1) * r * 0.8, cy + (j - 1) * r * 0.8
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    else:  # "more"
        for i in range(3):
            px = cx + (i - 1) * r * 0.8
            draw.ellipse((px - 3.5, cy - 3.5, px + 3.5, cy + 3.5), fill=color)


def _draw_element(draw: ImageDraw.ImageDraw, el: Element, w: int, h: int) -> None:
    box = _px(el.box, w, h)
    x0, y0, x1, y1 = box
    fg = el.fg or _TEXT
    if el.kind == "header":
        draw.rectangle(box, fill=el.fill)
        _text(
            draw, (x0 + 0.04 * w, (y0 + y1) / 2), el.label, el.font_size, fg, bold=True, anchor="lm"
        )
    elif el.kind == "footer":
        draw.rectangle(box, fill=el.fill)
    elif el.kind == "heading":
        _text(draw, (x0, y0), el.label, el.font_size, fg, bold=True)
    elif el.kind in ("text", "dense_text"):
        spacing = 3 if el.kind == "dense_text" else 8
        _text(draw, (x0, y0), el.label, el.font_size, fg, spacing=spacing)
    elif el.kind == "card":
        draw.rounded_rectangle(box, radius=12, fill="#FFFFFF", outline=_CARD_BORDER, width=2)
        _text(draw, (x0 + 22, y0 + 18), el.label, 20, _TEXT, bold=True)
    elif el.kind == "button":
        draw.rounded_rectangle(box, radius=10, fill=el.fill)
        _text(
            draw, ((x0 + x1) / 2, (y0 + y1) / 2), el.label, el.font_size, fg, bold=True, anchor="mm"
        )
    elif el.kind == "button_outline":
        draw.rounded_rectangle(box, radius=10, fill="#FFFFFF", outline=fg, width=2)
        _text(
            draw, ((x0 + x1) / 2, (y0 + y1) / 2), el.label, el.font_size, fg, bold=True, anchor="mm"
        )
    elif el.kind == "link":
        _text(draw, (x0, (y0 + y1) / 2), el.label, el.font_size, fg, anchor="lm")
    elif el.kind == "input":
        draw.rounded_rectangle(box, radius=8, fill="#FFFFFF", outline="#9AA3AF", width=2)
        _text(draw, (x0 + 14, (y0 + y1) / 2), el.label, el.font_size - 1, "#8A929C", anchor="lm")
    elif el.kind == "banner":
        draw.rounded_rectangle(box, radius=14, fill=el.fill)
        _text(
            draw,
            ((x0 + x1) / 2, (y0 + y1) / 2),
            el.label,
            el.font_size,
            fg,
            bold=True,
            anchor="mm",
            spacing=10,
        )
    elif el.kind == "image_card":
        draw.rounded_rectangle(box, radius=22, fill=el.fill)
        draw.rounded_rectangle((x0 + 30, y0 + 70, x0 + 90, y0 + 115), radius=6, fill="#D9B44A")
        _text(draw, (x0 + 30, y1 - 50), el.label, 22, fg, bold=True)
    elif el.kind == "icon":
        _draw_icon(draw, box, el.glyph or "more", fg)
    else:
        raise ValueError(f"Unknown element kind: {el.kind}")


def render_ui(spec: UISpec, watermark: str | None = None) -> Image.Image:
    """Render a UI spec to an RGB image of size ``(spec.width, spec.height)``."""
    img = Image.new("RGB", (spec.width, spec.height), spec.background)
    draw = ImageDraw.Draw(img)
    for el in spec.elements:
        _draw_element(draw, el, spec.width, spec.height)
    if watermark:
        _text(draw, (spec.width - 12, spec.height - 8), watermark, 12, "#B0B6BE", anchor="rd")
    return img


def save_ui(spec: UISpec, path: str | Path, watermark: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    render_ui(spec, watermark=watermark).save(p)
    return p
