import numpy as np, cv2
exec(open('prep3d.py').read().split("# ---------- background depth")[0])   # gives z, zf (person depth), m, pm, g, xx, yy, H, W
img = cv2.imread('src.jpg')
face = np.load('face_mask.npy').astype(np.uint8)
# ---- body/hair layer colour: fill what sits behind the face (hair at the sides, neck below)
inner = cv2.dilate(face, np.ones((13,13),np.uint8))
# behind the face: hair across each row (interpolated between the hair on both sides), neck towards the bottom
rowfill = img.astype(np.float32).copy()
for y in range(H):
    xs = np.where(inner[y])[0]
    if len(xs) == 0: continue
    a0, b0 = xs[0], xs[-1]
    lp = img[y, max(0,a0-30):max(1,a0-10)].reshape(-1,3); rp = img[y, b0+10:b0+30].reshape(-1,3)
    darkest = lambda P: P[np.argsort(P.sum(1))[:max(1,len(P)//2)]].mean(0)
    L = darkest(lp); R = darkest(rp)
    t = np.linspace(0, 1, b0-a0+1)[:,None]
    rowfill[y, a0:b0+1] = L*(1-t) + R*t
colfill = img.astype(np.float32).copy()
for x in range(W):
    ys = np.where(inner[:, x])[0]
    if len(ys) == 0: continue
    b0 = ys[-1]; colfill[ys, x] = img[min(H-1, b0+4), x]
wy = np.clip((yy - 600)/90, 0, 1)[...,None] * (np.abs(xx-800) < 120)[...,None]
wy = cv2.GaussianBlur(wy.astype(np.float32), (0,0), 15)[...,None] if wy.ndim == 2 else cv2.GaussianBlur(wy[...,0].astype(np.float32), (0,0), 15)[...,None]
structured = rowfill*(1-wy) + colfill*wy

sm = 0.5
small = cv2.resize(img, None, fx=sm, fy=sm, interpolation=cv2.INTER_AREA)
hs = cv2.resize(inner, None, fx=sm, fy=sm, interpolation=cv2.INTER_NEAREST)
fill = cv2.GaussianBlur(structured, (0,0), 2.5)
w = cv2.GaussianBlur(inner.astype(np.float32), (0,0), 2)[...,None]
body = (img*(1-w) + fill*w).astype(np.uint8)
cv2.imwrite('body.jpg', body, [cv2.IMWRITE_JPEG_QUALITY, 88])
# ---- face layer depth: surface of the face, extended past its outline, with a rounded falloff
fe = cv2.erode(face, np.ones((25,25),np.uint8)).astype(bool)
s4 = 4
zs = cv2.resize(zf, (W//s4, H//s4), interpolation=cv2.INTER_AREA); ks = cv2.resize(fe.astype(np.uint8), (W//s4, H//s4), interpolation=cv2.INTER_NEAREST).astype(bool)
f = zs.copy(); known = ks.copy()
for _ in range(80):
    k = cv2.dilate(np.where(known, f, -9).astype(np.float32), np.ones((3,3),np.uint8)); nk = (~known) & (k > -9); f[nk] = k[nk]; known |= nk
    if known.all(): break
f = cv2.GaussianBlur(f, (0,0), 2); fext = cv2.resize(f, (W,H), interpolation=cv2.INTER_CUBIC)
wf = cv2.GaussianBlur(fe.astype(np.float32), (0,0), 6)
zface = zf*wf + fext*(1-wf)
q = ((xx-800)/215)**2 + ((yy-462)/262)**2
zface -= 0.085*(1 - np.sqrt(np.clip(1-q, 0, 1)))
# ---- body layer depth: behind the face it is the back of the head / neck
zb_ = zf.copy()
fd = cv2.dilate(face, np.ones((9,9),np.uint8)).astype(bool)
zs = cv2.resize(zb_, (W//s4, H//s4), interpolation=cv2.INTER_AREA); ks = ~cv2.resize(fd.astype(np.uint8), (W//s4, H//s4), interpolation=cv2.INTER_NEAREST).astype(bool)
f = zs.copy(); known = ks.copy()
for _ in range(80):
    k = -cv2.dilate(np.where(known, -f, -9).astype(np.float32), np.ones((3,3),np.uint8)); nk = (~known) & (k < 9); f[nk] = k[nk]; known |= nk
    if known.all(): break
f = cv2.GaussianBlur(f, (0,0), 6); fill = cv2.resize(f, (W,H), interpolation=cv2.INTER_CUBIC)
behind = np.minimum(fill, zface - 0.07)
wb = cv2.GaussianBlur(fd.astype(np.float32), (0,0), 5)
zbody = zb_*(1-wb) + behind*wb
zbody = cv2.GaussianBlur(zbody, (0,0), 3.5)      # soften internal steps (hair over blazer etc.)
def save16(arr, name, lo=-1.0, hi=0.7, size=512):
    a = cv2.resize(arr, (size,size), interpolation=cv2.INTER_AREA)
    v = (np.clip((a - lo)/(hi - lo), 0, 1)*65535).astype(np.uint16); rgb = np.zeros((size,size,3), np.uint8)
    rgb[...,2] = (v >> 8).astype(np.uint8); rgb[...,1] = (v & 255).astype(np.uint8); cv2.imwrite(name, rgb)
save16(zbody, 'fg_depth.png'); save16(zface, 'face_depth.png')
print('face z center', zface[470,800], 'face edge', zface[470,630], 'behind-face', zbody[470,700], 'neck', zbody[760,800])
cv2.imwrite('body_prev.png', cv2.resize(body[100:850, 450:1150], (467,500)))
