import onnxruntime as ort, numpy as np, cv2
sess = ort.InferenceSession('/home/claude/models/depth_anything_v2_vits.onnx', providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]; print(inp.name, inp.shape)
img = cv2.cvtColor(cv2.imread('src.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)/255
def run(im, size=518):
    x = cv2.resize(im, (size,size), interpolation=cv2.INTER_CUBIC)
    x = (x - [0.485,0.456,0.406])/[0.229,0.224,0.225]
    x = x.transpose(2,0,1)[None].astype(np.float32)
    d = sess.run(None, {inp.name: x})[0]
    return np.squeeze(d)
shape = inp.shape
size = shape[2] if isinstance(shape[2], int) else 770
d = run(img, size)
# test-time flip augmentation
d2 = run(img[:, ::-1].copy(), size)[:, ::-1]
d = (d + d2)/2
d = cv2.resize(d, (1600,1600), interpolation=cv2.INTER_CUBIC)
print(d.min(), d.max())
np.save('depth_raw.npy', d)
v = ((d-d.min())/(d.max()-d.min())*255).astype(np.uint8)
cv2.imwrite('depth_vis.png', cv2.resize(v,(700,700)))
