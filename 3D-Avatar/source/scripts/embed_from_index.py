"""Rebuild index.html from source/template.html, reusing the images already embedded in index.html.

Use this after editing the template's HTML/CSS/JS. (build3d.py needs the original pipeline working
folder; the files in source/assets are not byte-identical to the images embedded in index.html.)

    python source/scripts/embed_from_index.py        (run from the 3D-Avatar folder)
"""
import base64
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
index, template = root / 'index.html', root / 'source' / 'template.html'
URI = re.compile(r'data:image/\w+;base64,[A-Za-z0-9+/=]+')
KEYS = ['__IMG__', '__BG__', '__MASK__', '__BODY__', '__FACEA__', '__FGZ__', '__BGZ__', '__FACEZ__']

uris = URI.findall(index.read_text(encoding='utf-8'))
tpl = template.read_text(encoding='utf-8')
order = sorted(KEYS, key=tpl.index)          # placeholders in the order they appear in the page
if len(uris) != len(KEYS) or any(tpl.count(f'"{k}"') != 1 for k in KEYS):
    sys.exit(f'expected {len(KEYS)} embedded images and one of each placeholder; found {len(uris)} images')
embedded = dict(zip(order, uris))
# layers rebuilt locally replace the embedded ones (see fill_behind_face.py)
for key, path in {'__BODY__': root / 'source' / 'assets' / 'body_filled.jpg',
                  '__FGZ__': root / 'source' / 'assets' / 'fg_depth_filled.png',
                  '__FACEA__': root / 'source' / 'assets' / 'face_alpha_grown.jpg'}.items():
    if path.exists():
        mime = 'png' if path.suffix == '.png' else 'jpeg'
        embedded[key] = f'data:image/{mime};base64,' + base64.b64encode(path.read_bytes()).decode()
for k, u in embedded.items():
    tpl = tpl.replace(f'"{k}"', f'"{u}"')
index.write_text(tpl, encoding='utf-8', newline='\n')
print(f'wrote {index} ({index.stat().st_size:,} bytes)')
