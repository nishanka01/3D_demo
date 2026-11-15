import base64
enc = lambda f, t='jpeg': f'data:image/{t};base64,' + base64.b64encode(open(f,'rb').read()).decode()
s = open('template3d.html').read()
for k, v in {'__IMG__': enc('src.jpg'), '__BG__': enc('bg.jpg'), '__MASK__': enc('fg_mask.jpg'), '__FGZ__': enc('fg_depth.png','png'), '__BGZ__': enc('bg_depth.png','png'), '__BODY__': enc('body.jpg'), '__FACEA__': enc('face_alpha.jpg'), '__FACEZ__': enc('face_depth.png','png')}.items():
    s = s.replace(k, v)
open('index.html','w').write(s)
