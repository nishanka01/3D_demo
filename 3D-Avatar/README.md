# 3D Avatar

A live, interactive 3D version of the avatar image: it blinks, glances around, nods, winks and talks. The view stays on the front; it does not turn or follow the cursor.

## Open it

Double-click **index.html**. It opens in any modern browser (Chrome, Edge, Firefox).

- The file is self-contained: all images and depth maps are embedded inside it.
- It needs an internet connection the first time, because it loads the three.js 3D library from cdn.jsdelivr.net.

### Controls
- **Tap/click her** for a random reaction.
- **Buttons:** Blink, Wink, Look around, Nod, No, Say hi (talks with mouth movement and voice).
- **Keyboard:** B blink · W wink · L look around · N nod · X no · H say hi.

## Folder contents

| Path | What it is |
|---|---|
| `index.html` | The finished avatar page (final version). |
| `previous-versions/procedural-3d-model.html` | The first attempt: a hand-modelled 3D character (full 360° but not identical to the image). |
| `source/template.html` | The page source before the images are embedded (placeholders like `__IMG__`). |
| `source/assets/avatar.jpg` | The original avatar image. |
| `source/assets/person_mask.png`, `fg_mask.jpg` | Cut-out mask of the person. |
| `source/assets/face_alpha.jpg` | Cut-out mask of the face layer. |
| `source/assets/bg.jpg` | Background plate with the person removed and filled in. |
| `source/assets/body.jpg` | Hair/body layer with the area behind the face filled in. |
| `source/assets/*_depth.png` | 16-bit depth maps (packed in the red and green channels) for the background, body and face layers. |
| `source/scripts/` | The Python pipeline that produced the assets. |

## Editing the page

`source/scripts/fill_behind_face.py` rebuilds the hidden area of the hair/body layer behind the face and grows the face cut-out over the cheek's shadow rim. The results are `body_filled.jpg`, `fg_depth_filled.png` and `face_alpha_grown.jpg` in `source/assets`, and they stop dark smears and stretched hair from showing beside the cheek and jaw when she turns. Run it before the step below if you change those layers.

Edit `source/template.html`, then run `python source/scripts/embed_from_index.py` from this folder. It rebuilds `index.html` from the template and keeps the images already embedded in it. (The files in `source/assets` are not byte-identical to the embedded images.) The version before the mobile layout is kept in `previous-versions/index-before-mobile.html`.

To serve it locally: `python -m http.server 8000`, then open http://localhost:8000.

## Rebuilding from source (optional)

The scripts were run in one working folder in this order. Copy `avatar.jpg` there as `src.jpg`, `person_mask.png` as `mask_u2net_human_seg.png`, and `template.html` as `template3d.html`.

1. `depth.py`: estimates depth with Depth Anything V2 (ONNX model `depth_anything_v2_vits.onnx`).
2. `prep.py`: builds the background plate and person mask.
3. `prep3d.py`: converts depth into 3D layer depths.
4. `facemask.py`: cuts out the face.
5. `prep_layers.py`: builds the body/hair layer and the face depth.
6. `build3d.py`: embeds everything into `index.html`.

The person mask was made with `rembg` (model `u2net_human_seg`).

Python packages: `numpy opencv-python-headless onnxruntime rembg pillow`.
