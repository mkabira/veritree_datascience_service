"""
Regenerate api/static/favicon.ico from the veritree logo.

Run after changing docs/assets/vt_logo.jpeg:

    python3 scripts/generate_favicon.py
"""

import os

from PIL import Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'docs', 'assets', 'vt_logo.jpeg')
TARGET = os.path.join(ROOT, 'api', 'static', 'favicon.ico')

# An .ico embeds several resolutions and the browser picks per context (tab,
# bookmark, retina). Capped at the 225px source so nothing is upscaled.
SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)]


def main():
    """
    Rebuild the .ico from the logo and write it into api/static.

    :return: None
    :raises SystemExit: if the source logo is not square
    """

    source = Image.open(SOURCE).convert('RGBA')

    if source.width != source.height:
        raise SystemExit(f"{SOURCE} is {source.size}; favicons must be square")

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    source.save(TARGET, format='ICO', sizes=SIZES)

    print(f"wrote {os.path.relpath(TARGET, ROOT)} with sizes {SIZES}")


if __name__ == '__main__':
    main()
