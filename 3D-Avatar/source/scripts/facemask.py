import numpy as np, cv2
img = cv2.imread('src.jpg'); H, W = img.shape[:2]
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
L = lab[...,0]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
cand = (L > 95).astype(np.uint8)
region = (((xx-800)/235)**2 + ((yy-450)/300)**2 < 1).astype(np.uint8)
cand &= region
n, lbl = cv2.connectedComponents(cand)
face = (lbl == lbl[470, 800]).astype(np.uint8)
# fill holes (eyes, brows, lips)
ff = face.copy(); h2 = np.zeros((H+2, W+2), np.uint8); cv2.floodFill(ff, h2, (0,0), 1); face = face | (1 - ff)
# jaw line: cut off the neck with the face ellipse below the cheeks
jaw = (((xx-800)/196)**2 + ((yy-462)/241)**2 < 1).astype(np.uint8)
face = np.where(yy > 560, face & jaw, face).astype(np.uint8)
face = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
n, lbl, stats, _ = cv2.connectedComponentsWithStats(face); face = (lbl == lbl[470,800]).astype(np.uint8)
cnts,_ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
hull = cv2.convexHull(max(cnts, key=cv2.contourArea)); face = np.zeros_like(face); cv2.fillPoly(face, [hull], 1)
face = np.where(yy > 560, face & jaw, face).astype(np.uint8)
np.save('face_mask.npy', face)
soft = cv2.GaussianBlur(face.astype(np.float32), (0,0), 1.6)
cv2.imwrite('face_alpha.jpg', (soft*255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
vis = img.copy(); vis[face==0] = (vis[face==0]*0.25).astype(np.uint8)
cv2.imwrite('face_vis.png', cv2.resize(vis[100:800, 450:1150], (560,560)))
print('face px', face.sum(), 'bbox', cv2.boundingRect(face))
