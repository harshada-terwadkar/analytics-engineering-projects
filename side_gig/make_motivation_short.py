#!/usr/bin/env python3
# MoviePy 2.2.1 compatible (no vfx/editor). Ultra-large readable text + exact background.

import os, argparse
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from moviepy import ImageClip, VideoClip, AudioFileClip, CompositeVideoClip, afx

# Canvas
W, H = 1080, 1920          # vertical 9:16
SAFE = 24                  # tighter padding to allow bigger text

# Fonts (Pillow will fall back to default if missing)
FONT_REGULAR = "DejaVuSans.ttf"
FONT_BOLD    = "DejaVuSans-Bold.ttf"

# -------------------- compatibility shims --------------------
def set_duration_safe(clip, duration):
    return clip.with_duration(duration) if hasattr(clip, "with_duration") else clip.set_duration(duration)

def set_position_safe(clip, pos):
    return clip.with_position(pos) if hasattr(clip, "with_position") else clip.set_position(pos)

def set_audio_safe(clip, audio):
    return clip.with_audio(audio) if hasattr(clip, "with_audio") else clip.set_audio(audio)

def audio_slice(aclip, start, end):
    if hasattr(aclip, "subclipped"): return aclip.subclipped(start, end)
    if hasattr(aclip, "time_slice"): return aclip.time_slice(start, end)
    return aclip.subclip(start, end)

# -------------------- helpers --------------------
def letterbox_to_canvas(img: Image.Image, target_w: int, target_h: int, fill=(0,0,0)) -> Image.Image:
    """Fit entire image into canvas without cropping; add bars if needed (exact look preserved)."""
    iw, ih = img.size
    scale = min(target_w/iw, target_h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    x = (target_w - nw)//2
    y = (target_h - nh)//2
    canvas.paste(resized, (x, y))
    return canvas

def try_font(path: Optional[str], size: int):
    try:
        return ImageFont.truetype(path or FONT_BOLD, size)
    except:
        return ImageFont.load_default()

def multiline_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, spacing: int, stroke: int):
    left, top, right, bottom = draw.multiline_textbbox((0,0), text, font=font, spacing=spacing, stroke_width=stroke, align="center")
    return right-left, bottom-top

# --- NEW: aggressive max-fit text packer ---
def maxfit_text_image(
    text: str,
    box_size: Tuple[int,int],
    font_path: Optional[str],
    min_px: int,
    max_px: int,
    *,
    max_lines: int = 3,           # try 2–3 lines first → bigger letters
    spacing: int = 0,
    stroke: int = 3,
    fill=(255,255,255,255),
    shadow=(0,0,0,200),
    panel_alpha: int = 0,         # 0 = no panel; all space for letters
    panel_radius: int = 0,
    padding: int = 0              # keep 0 to maximize area
) -> Image.Image:
    """
    Aggressive text fitter: packs into as few lines as possible (up to max_lines),
    choosing the largest font that fits both width and height.
    """
    BW, BH = int(box_size[0]), int(box_size[1])
    probe = Image.new("RGBA", (BW, BH), (0,0,0,0))
    draw = ImageDraw.Draw(probe)

    def text_size(lines, font):
        s = "\n".join(lines)
        l,t,r,b = draw.multiline_textbbox((0,0), s, font=font, spacing=spacing, stroke_width=stroke, align="center")
        return (r-l, b-t)

    def wrap_greedy(words, font, target_lines):
        # Greedy partition into <= target_lines lines
        lines = ["" for _ in range(target_lines)]
        idx = 0
        for w in words:
            test = (lines[idx] + " " + w).strip()
            tw, _ = text_size([test], font)
            if tw <= BW - 2*padding:
                lines[idx] = test
            else:
                if idx < target_lines - 1:
                    idx += 1
                    lines[idx] = w
                else:
                    # force append to last; will fail fit and shrink font
                    lines[idx] = (lines[idx] + " " + w).strip()
        return [ln for ln in lines if ln]

    words = text.split()

    best_img = None
    # Try fewer lines first (2, then max_lines), then 4 as last resort
    for lines_cap in [2, max_lines, 4]:
        lo, hi = int(min_px), int(max_px)
        chosen = None
        chosen_lines = None
        while lo <= hi:
            mid = (lo + hi) // 2
            font = try_font(font_path, mid)
            lines = wrap_greedy(words, font, lines_cap)
            tw, th = text_size(lines, font)
            if tw <= BW - 2*padding and th <= BH - 2*padding and len(lines) <= lines_cap:
                chosen, chosen_lines = (mid, lines)
                lo = mid + 1  # try bigger
            else:
                hi = mid - 1  # too big → smaller
        if chosen:
            # render with chosen
            font = try_font(font_path, chosen)
            s = "\n".join(chosen_lines)
            tw, th = text_size(chosen_lines, font)

            out = Image.new("RGBA", (BW, BH), (0,0,0,0))

            # optional minimal panel (disabled by default)
            if panel_alpha > 0:
                pw = min(BW, tw + 2*padding)
                ph = min(BH, th + 2*padding)
                panel = Image.new("RGBA", (pw, ph), (0,0,0,0))
                pdraw = ImageDraw.Draw(panel)
                pdraw.rounded_rectangle([0,0,pw,ph], radius=int(panel_radius), fill=(0,0,0,int(panel_alpha)))
                panel = panel.filter(ImageFilter.GaussianBlur(1))
                out.alpha_composite(panel, ((BW - pw)//2, (BH - ph)//2))

            tx = int((BW - tw)//2)
            ty = int((BH - th)//2)
            d2 = ImageDraw.Draw(out)
            # shadow then fill
            d2.multiline_text((tx+2, ty+2), s, font=font, fill=shadow, spacing=spacing, align="center", stroke_width=stroke)
            d2.multiline_text((tx,   ty),   s, font=font, fill=fill,   spacing=spacing, align="center", stroke_width=stroke)
            best_img = out
            break
    if best_img is None:
        # ultra fallback
        font = try_font(font_path, int(min_px))
        tw, th = text_size([text], font)
        out = Image.new("RGBA", (BW, BH), (0,0,0,0))
        tx = int((BW - tw)//2)
        ty = int((BH - th)//2)
        d2 = ImageDraw.Draw(out)
        d2.text((tx+2,ty+2), text, font=font, fill=shadow, stroke_width=stroke)
        d2.text((tx,ty),     text, font=font, fill=(255,255,255,255), stroke_width=stroke)
        return out
    return best_img

# -------------------- main --------------------
def build_video(
    quote: str,
    author: str,
    audio_path: str,
    audio_start: float,
    duration: float,
    out_path: str,
    bg_image_path: str,
    fps: int = 24
):
    if not os.path.exists(bg_image_path):
        raise FileNotFoundError(f"Background not found: {bg_image_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # 1) Background EXACT (letterboxed, no crop/stretch)
    bg_src = Image.open(bg_image_path).convert("RGB")
    bg = letterbox_to_canvas(bg_src, W, H, fill=(0,0,0))
    bg_clip = set_duration_safe(ImageClip(np.array(bg)), duration)

    # 2) Ultra-large, readable text overlays
    quote_box = (W - 2 * SAFE, int(H * 0.86))  # lots of vertical room
    author_box_h = 360                          # larger author area

    # QUOTE (target 2–3 lines, very big)
    quote_img = maxfit_text_image(
        quote, quote_box, font_path=FONT_BOLD,
        min_px=120, max_px=2600, max_lines=3, spacing=0, stroke=4,
        panel_alpha=0, padding=0
    )
    quote_clip = set_duration_safe(ImageClip(np.array(quote_img)), duration)
    # Slightly high placement (poster feel)
    quote_clip = set_position_safe(quote_clip, ('center', int(H*0.34 - quote_box[1]//2)))

    # AUTHOR (often 1–2 lines, also big)
    author_img = maxfit_text_image(
        author, (W - 2 * SAFE, author_box_h), font_path=FONT_BOLD,
        min_px=120, max_px=1600, max_lines=2, spacing=0, stroke=3,
        panel_alpha=0, padding=0
    )
    author_clip = set_duration_safe(ImageClip(np.array(author_img)), duration)
    author_clip = set_position_safe(author_clip, ('center', H - SAFE - author_box_h))

    # 3) Audio slice + fades + volume
    raw_audio = AudioFileClip(audio_path)
    audio = audio_slice(raw_audio, audio_start, audio_start + duration)
    try:
        audio = afx.audio_fadein(audio, 0.4)
        audio = afx.audio_fadeout(audio, 0.6)
        audio = afx.volumex(audio, 0.9)
    except Exception:
        if hasattr(audio, "audio_fadein"):  audio = audio.audio_fadein(0.4)
        if hasattr(audio, "audio_fadeout"): audio = audio.audio_fadeout(0.6)
        if hasattr(audio, "volumex"):       audio = audio.volumex(0.9)

    # 4) Compose + export
    final = CompositeVideoClip([bg_clip, quote_clip, author_clip])
    final = set_audio_safe(final, audio)
    final.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="3000k",
        threads=4
    )

def parse_args():
    ap = argparse.ArgumentParser(description="Cinematic motivational short (ultra-large text, exact background).")
    ap.add_argument("--quote", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--audio", required=True, help="Path to music (mp3/wav)")
    ap.add_argument("--audio_start", type=float, default=65.0, help="Start time in seconds")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--bg", required=True, help="Path to your exact sunrise image")
    ap.add_argument("--out", default="motivation_short.mp4")
    ap.add_argument("--fps", type=int, default=24)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_video(
        quote=args.quote,
        author=args.author,
        audio_path=args.audio,
        audio_start=args.audio_start,
        duration=args.duration,
        out_path=args.out,
        bg_image_path=args.bg,
        fps=args.fps
    )
