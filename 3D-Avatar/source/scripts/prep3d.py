import numpy as np, cv2
d = np.load('depth_raw.npy').astype(np.float32)          # relative disparity, bigger = nearer
m = cv2.imread('mask_u2net_human_seg.png', 0).astype(np.float32)/255
H, W = d.shape
# disparity -> world z (towards camera), shared by both layers
kd = np.array([0.15, 0.3, 1.0, 1.5, 2.3, 3.06, 4.3, 6.9])
kz = np.array([-0.95, -0.85, -0.45, -0.24, -0.08, 0.07, 0.30, 0.55])
z = np.interp(d, kd, kz).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
g = lambda cx, cy, sx, sy: np.exp(-(((xx-cx)/sx)**2 + ((yy-cy)/sy)**2))
# ---------- person depth ----------
pm = (m > 0.5).astype(np.uint8)
zf = z.copy()
# extend the person depth outward so the soft alpha edge never drops to the wall depth
small = cv2.resize(zf, (400,400)); sm = cv2.erode(cv2.resize(pm, (400,400), interpolation=cv2.INTER_NEAREST), np.ones((9,9),np.uint8))
hole = (1 - cv2.erode(sm, np.ones((3,3),np.uint8))).astype(np.uint8)
ext = cv2.inpaint(((small+1)*30000).astype(np.uint16).astype(np.float32).astype(np.uint8) if False else small, hole, 8, cv2.INPAINT_TELEA) if False else None
# simple iterative dilation-based extension (float safe)
f = small.copy(); known = sm.astype(bool)
for _ in range(60):
    k = cv2.dilate(np.where(known, f, -9).astype(np.float32), np.ones((5,5),np.uint8))
    nk = (~known) & (k > -9)
    f[nk] = k[nk]; known = known | nk
    if known.all(): break
f = cv2.GaussianBlur(f, (0,0), 3)
fext = cv2.resize(f, (W,H), interpolation=cv2.INTER_CUBIC)
w = cv2.GaussianBlur(cv2.erode(pm, np.ones((31,31),np.uint8)).astype(np.float32), (0,0), 5)
zf = zf*w + fext*(1-w)
# face detail: local relief boost + sculpted nose, eye sockets, lips, cheeks
face = g(800, 470, 190, 235)**2
hp = zf - cv2.GaussianBlur(zf, (0,0), 18)
zf += 1.2*hp*face
zf += 0.038*g(800, 548, 20, 34) + 0.012*g(800, 500, 13, 40)        # nose tip + bridge
zf -= 0.016*(g(707, 444, 42, 26) + g(893, 444, 42, 26))           # eye sockets
zf += 0.010*g(800, 612, 48, 16)                                    # lips
zf += 0.012*(g(670, 540, 55, 45) + g(930, 540, 55, 45))           # cheeks
zf -= 0.02*g(800, 690, 120, 30)*(1-face)                           # under-chin recess
zf = np.maximum(zf, -0.30)
zf = cv2.GaussianBlur(zf, (0,0), 1.5)
# ---------- background depth (person removed) ----------
N = 400
bs = cv2.resize(z, (N,N), interpolation=cv2.INTER_AREA)
hole = cv2.dilate(cv2.resize(pm, (N,N), interpolation=cv2.INTER_NEAREST), np.ones((17,17),np.uint8)).astype(bool)
rows = np.arange(N)[:,None].repeat(N,1)*(1600/N)
deskTop = 1418
b = bs.copy()
# behind her (above the desk) = the far wall; on the desk rows, continue the desk from below
wall = np.median(bs[(~hole) & (rows < 900)])
b[hole & (rows < deskTop)] = wall
known = ~(hole & (rows >= deskTop))
for _ in range(80):
    up = np.roll(b, -1, axis=0); kup = np.roll(known, -1, axis=0)
    nk = (~known) & kup
    b[nk] = up[nk]; known |= nk
    if known.all(): break
b = cv2.GaussianBlur(b, (0,0), 10)
# keep the desk edge crisp-ish: re-blend the desk band with a lighter blur
desk = cv2.GaussianBlur(np.where(rows >= deskTop-8, b, b), (0,0), 3)
zb = cv2.resize(b, (W,H), interpolation=cv2.INTER_CUBIC)
zb = np.clip(zb, -0.95, 0.6)
rowsF = np.arange(H)[:,None].astype(np.float32).repeat(W,1)
flat = -0.9 + 0.3*(zb + 0.9)
wdesk = np.clip((rowsF - 1395)/35, 0, 1)
zb = flat*(1-wdesk) + zb*wdesk
zb = cv2.GaussianBlur(zb, (0,0), 8)
def save16(arr, name, lo=-1.0, hi=0.7, size=512):
    a = cv2.resize(arr, (size,size), interpolation=cv2.INTER_AREA)
    v = np.clip((a - lo)/(hi - lo), 0, 1)*65535
    v = v.astype(np.uint16); rgb = np.zeros((size,size,3), np.uint8)
    rgb[...,2] = (v >> 8).astype(np.uint8); rgb[...,1] = (v & 255).astype(np.uint8)   # cv2 is BGR: R=high, G=low
    cv2.imwrite(name, rgb)
save16(zf, 'fg_depth.png'); save16(zb, 'bg_depth.png')
print('fg z range', zf[pm>0].min(), zf[pm>0].max(), 'face', zf[470,800], 'nose', zf[552,800], 'hands', zf[1380,800], 'bg', np.percentile(zb,[5,50,95]))
vis = ((np.clip((zf+0.4)/1.0,0,1))*255).astype(np.uint8)
cv2.imwrite('zf_vis.png', cv2.resize(cv2.applyColorMap(vis, cv2.COLORMAP_JET)*pm[...,None], (600,600)))
