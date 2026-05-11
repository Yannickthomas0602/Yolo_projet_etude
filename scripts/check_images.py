from pathlib import Path
import cv2
from PIL import Image

root = Path('dataset_oiseaux')
errors = []
for p in root.rglob('*'):
    if p.is_file():
        if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:
            continue
        try:
            img = cv2.imread(str(p))
            if img is None:
                # try PIL
                try:
                    Image.open(p).verify()
                    # PIL can open but cv2 can't, still consider OK
                except Exception:
                    errors.append(str(p))
        except Exception:
            errors.append(str(p))

if errors:
    print('INVALID_IMAGES_FOUND')
    for e in errors:
        print(e)
else:
    print('NO_ERRORS')
    print('checked files count:', sum(1 for _ in root.rglob('*') if _.is_file()))
