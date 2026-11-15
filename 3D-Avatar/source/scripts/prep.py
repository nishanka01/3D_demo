import numpy as np, cv2
img = cv2.imread('src.jpg')                       # BGR
m = cv2.imread('mask_u2net_human_seg.png', 0).astype(np.float32)/255
H, W = m.shape
# ---- foreground alpha: tighten slightly, soft edge
a = cv2.GaussianBlur(np.clip((m-0.25)/0.6,0,1), (0,0), 1.2)
cv2.imwrite('fg_mask.jpg', (a*255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
# ---- background: remove the person generously, then mirror-fill each row
hole = (cv2.dilate((m>0.04).astype(np.uint8), np.ones((51,51),np.uint8)) > 0)
out = img.copy().astype(np.float32)
# copy from the same row shifted by whole multiples of the wall-slat period, so slats stay straight
P = 87
todo = hole.copy(); xs = np.arange(W)
for k in range(1, 20):
    for sgn in (-1, 1):
        off = sgn*k*P
        src = xs + off
        ok = (src >= 0) & (src < W)
        for y in np.where(todo[:780].any(1))[0]:
            r = todo[y]
            cand = r & ok
            sx = np.clip(src, 0, W-1)
            good = cand & ~hole[y, sx]
            if good.any():
                out[y, good] = img[y, sx[good]]
                todo[y, good] = False
    if not todo.any(): break
# lower rows (credenza, desk): extend the nearest outside column horizontally (keeps horizontal edges straight)
for y in range(780, H):
    r = hole[y]
    if not r.any(): continue
    x = 0
    while x < W:
        if not r[x]: x += 1; continue
        a0 = x
        while x < W and r[x]: x += 1
        b0 = x - 1
        lv = out[y, a0-1] if a0 > 0 else None; rv = out[y, b0+1] if b0+1 < W else None
        if lv is None: lv = rv
        if rv is None: rv = lv
        L = b0 - a0 + 1; tt = np.clip((np.arange(L) - L/2)/max(20, L*0.25) + 0.5, 0, 1)[:,None]
        out[y, a0:b0+1] = lv*(1-tt) + rv*tt
out = out.astype(np.uint8)
# soften the filled area a little, blend back with a soft edge
blur = cv2.GaussianBlur(out, (0,0), 2.5)
hm = cv2.GaussianBlur(hole.astype(np.float32), (0,0), 6)[...,None]
fill = out*(1-hm) + blur*hm
tight = (cv2.dilate((m>0.08).astype(np.uint8), np.ones((41,41),np.uint8)) > 0).astype(np.float32)
tm = cv2.GaussianBlur(tight, (0,0), 3)[...,None]
bg = (img.astype(np.float32)*(1-tm) + fill*tm).astype(np.uint8)
cv2.imwrite('bg.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('done')
