"""Rebuild the part of the hair/body layer that sits behind the face.

When the avatar turns, the face layer slides sideways and uncovers a strip of the body layer next to the
cheek and under the jaw. The old fill there was a smeared black blob and a blurred neck, which showed up
as dark smears and stretched hair. This replaces it by mirroring the real hair / neck / skin just outside
the face edge back inward (nearest edge wins, blended so there are no seams), with a soft contact shadow
next to the face.

Reads the original layers embedded in previous-versions/index-before-mobile.html and writes
source/assets/body_filled.jpg and fg_depth_filled.png, which embed_from_index.py then embeds.

    python source/scripts/fill_behind_face.py        (run from the 3D-Avatar folder)
"""
import base64
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

root = Path(__file__).resolve().parents[2]
# always start from the original layers (index.html itself already carries the rebuilt ones)
html = (root / 'previous-versions' / 'index-before-mobile.html').read_text(encoding='utf-8')
uris = re.findall(r'data:image/\w+;base64,([A-Za-z0-9+/=]+)', html)
load = lambda i, mode: Image.open(io.BytesIO(base64.b64decode(uris[i]))).convert(mode)
# embedded order: IMG, BG, MASK, BODY, FACEA, FGZ, BGZ, FACEZ
src = np.asarray(load(0, 'RGB'), np.float32)
person = np.asarray(load(2, 'L'), np.float32) / 255
face = np.asarray(load(4, 'L'), np.float32) / 255
face0 = face.copy()
H, W = face.shape

# ---- grow the face cut-out over the shaded skin rim it misses (up to 8 px, mostly on her right cheek).
# Left behind on the body layer, that rim showed as a brown curve beside the cheek when she turns.
lum = src.mean(2)
skin = (lum > 55) & (src[..., 0] > src[..., 2] + 25)

# the neck's own width, measured on the visible neck below the chin
NROW = 770
nk = (src[NROW, :, 0] > 110) & (lum[NROW] > 80)
nl = nr = 800
while nl > 0 and nk[nl - 1]: nl -= 1
while nr < W - 1 and nk[nr + 1]: nr += 1
print(f'neck spans x {nl}..{nr} at y {NROW}')

# grow over the skin rim along the cheeks and down the jaw sides, but never across the neck under the chin
Yi, Xi = np.mgrid[0:H, 0:W]
skin &= (Yi < 720) & ((Yi < 640) | (Xi < nl) | (Xi > nr))
grown = face > 0.5
for _ in range(8):
    ring = np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 127
    grown |= ring & skin
soft = np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
face = np.maximum(face, soft)
Image.fromarray((face * 255).round().astype(np.uint8), 'L').save(root / 'source' / 'assets' / 'face_alpha_grown.jpg', quality=95)

# region to rebuild: the face cut-out, grown a few px so its soft edge is covered too
R = np.asarray(Image.fromarray((face > 0.03).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))) > 127
# Under the chin, keep the original pixels where the chin's soft edge is mostly neck: the chin shows the
# shadow crease on the neck through it, and that crease is what puts the chin in front of the neck
# (replacing it made the neck look pushed forward). Where the edge is mostly chin it is still replaced,
# or a pale outline of the chin stays on the neck when she turns.
R[620:, nl:nr + 1] &= face0[620:, nl:nr + 1] > 0.3       # across the neck only; the jaw sides stay rebuilt
GAP = 3                                     # skip the anti-aliased rim just outside the face edge
MAXD = 60                                   # turning uncovers well under this; deeper pixels never show


def runs(m):
    """Length of the run of True ending at each pixel, scanning along axis 1 (left -> right)."""
    out = np.zeros(m.shape, np.int32)
    for x in range(m.shape[1]):
        out[:, x] = (out[:, x - 1] + 1) * m[:, x] if x else m[:, x]
    return out


dl = runs(R)                                # distance to the edge on the left
dr = runs(R[:, ::-1])[:, ::-1]              # ... right
du = runs(R.T).T                            # ... above
dd = runs(R.T[:, ::-1])[:, ::-1].T          # ... below

ys, xs = np.nonzero(R)

NECK_L, NECK_R = nl + 6, nr - 6                 # only that strip is neck behind the jaw
cands = [  # (distance, source y, source x) for each mirror direction
    (dl[ys, xs], ys, xs - 2 * dl[ys, xs] + 1 - GAP),
    (dr[ys, xs], ys, xs + 2 * dr[ys, xs] - 1 + GAP),
    (du[ys, xs], ys - 2 * du[ys, xs] + 1 - GAP, xs),
    (dd[ys, xs], ys + 2 * dd[ys, xs] - 1 + GAP, xs),
]
acc = np.zeros((len(ys), 3), np.float32)
wsum = np.zeros(len(ys), np.float32)
for i, (d, sy, sx) in enumerate(cands):
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & (d <= MAXD)
    sy, sx = np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1)
    ok &= ~R[sy, sx] & (person[sy, sx] > 0.6)
    if i < 3:   # sideways / upward: behind a cheek is hair, so only mirror hair (no skin, no earring)
        ok &= (src[sy, sx, 0] < 115) & (lum[sy, sx] < 85)
    w = ok / np.maximum(d, 1).astype(np.float32) ** 4       # nearest edge dominates, seams blend
    if i == 3:
        # behind the lower face is the neck: where there is neck skin below, it wins outright (camera turns
        # uncover this area right above the visible neck, so anything else reads as a dark patch on it);
        # elsewhere (below the jaw corners is hair shadow) the sides win, or the jaw sides go grey
        neck = (src[sy, sx, 0] > 110) & (lum[sy, sx] > 80) & (ys > 560) & (xs >= NECK_L) & (xs <= NECK_R)
        w *= np.where(neck, 1e6, 0.08)
    acc += src[sy, sx] * w[:, None]
    wsum += w

# deep inside (never uncovered) and anything without a valid source: dark hair colour
hair = src[(person > 0.9) & ~R & (src.mean(2) < 40)].mean(0)
fill = np.where(wsum[:, None] > 0, acc / np.maximum(wsum, 1e-12)[:, None], hair)

# soft contact shadow: what sits right behind the face edge is a little darker
dmin = np.minimum(np.minimum(dl[ys, xs], dr[ys, xs]), np.minimum(du[ys, xs], dd[ys, xs])).astype(np.float32)
t = np.clip(dmin / 14, 0, 1)
shade = 1 - 0.22 * t * t * (3 - 2 * t)
# neck behind the jaw: the skin carries on from the visible neck, then falls into the jaw's shadow the
# further up under the jaw it goes, so a turn shows the jaw passing in front of a shaded neck
neck_px = (dd[ys, xs] <= MAXD) & (wsum > 1e3)
tj = np.clip(dd[ys, xs] / 30, 0, 1)
shade = np.where(neck_px, 1 - 0.12 * tj * tj * (3 - 2 * tj), shade)
fill *= shade[:, None]

out = src.copy()
out[ys, xs] = fill
# smooth the rebuilt area slightly and feather it into the untouched pixels
blur = np.asarray(Image.fromarray(out.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), np.float32)
k = np.asarray(Image.fromarray(R.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5)), np.float32)[..., None] / 255
out = np.where(R[..., None], blur, out) * k + src * (1 - k)

dst = root / 'source' / 'assets' / 'body_filled.jpg'
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(dst, quality=92, optimize=True)
print(f'wrote {dst} ({dst.stat().st_size:,} bytes), rebuilt {len(ys):,} px')

# ---- depth of the same hidden area -----------------------------------------------------------------
# The body layer's depth behind the face was a deep pit (down to -0.25 while the hair around it sits near
# -0.05). Turning showed the pit's steep walls with the hair texture stretched across them. Replace it
# with a smooth surface that continues the surrounding hair/neck depth (Laplace fill from the edge).
LO, HI = -1.0, 0.7                                   # must match decodeDepth() in the page
zimg = np.asarray(load(5, 'RGB'), np.float32)
Z = LO + (zimg[..., 0] * 256 + zimg[..., 1]) / 65535 * (HI - LO)
h, w = Z.shape
Rz = np.asarray(Image.fromarray(R.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)
                .filter(ImageFilter.MaxFilter(5))) > 20
Zf = Z.copy()
Zf[Rz] = Z[~Rz & np.asarray(Image.fromarray(Rz.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7)), bool)].mean()
for _ in range(3000):                                # Jacobi iterations; the edge values stay fixed
    avg = (np.roll(Zf, 1, 0) + np.roll(Zf, -1, 0) + np.roll(Zf, 1, 1) + np.roll(Zf, -1, 1)) / 4
    Zf[Rz] = avg[Rz]
# never let the hidden surface sit far behind the face in front of it: the further back it is, the more a
# camera turn slides it out from under the jaw/cheek (that was the dark patch below the jaw)
fimg = np.asarray(load(7, 'RGB'), np.float32)
FZ = LO + (fimg[..., 0] * 256 + fimg[..., 1]) / 65535 * (HI - LO)
FZ = np.asarray(Image.fromarray(FZ).resize((w, h), Image.BILINEAR)) if FZ.shape != Z.shape else FZ
# Only for the cheeks/temples: behind the jaw the neck must stay at neck depth, behind the chin, or the neck
# looks pushed forward when she turns (the jaw has to pass in front of it).
yy = (np.arange(h)[:, None] + 0.5) * (H / h)
upper = np.clip((640 - yy) / 80, 0, 1) * np.ones((1, w))
Zf = np.where(Rz, np.maximum(Zf, (FZ - 0.04) * upper + Zf * (1 - upper)), Zf)
# blend over a couple of pixels so the edge has no crease
k = np.asarray(Image.fromarray(Rz.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
Zf = Zf * k + Z * (1 - k)
q = np.round((Zf - LO) / (HI - LO) * 65535).clip(0, 65535).astype(np.uint32)
enc = np.stack([q >> 8, q & 255, np.zeros_like(q)], -1).astype(np.uint8)
zdst = root / 'source' / 'assets' / 'fg_depth_filled.png'
Image.fromarray(enc, 'RGB').save(zdst, optimize=True)
print(f'wrote {zdst}; hidden depth range was {Z[Rz].min():+.2f}..{Z[Rz].max():+.2f}, now {Zf[Rz].min():+.2f}..{Zf[Rz].max():+.2f}')
