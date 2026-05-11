from pathlib import Path
import os
import shutil

root = Path('dataset_oiseaux')
out = Path('dataset_train_only')

if out.exists():
    print('Removing existing', out)
    shutil.rmtree(out)

for split in ['train', 'val']:
    for cls in (root / 'train').iterdir():
        if not cls.is_dir():
            continue
        dest_dir = out / split / cls.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        # hardlink all files from source train class into dest_dir
        for f in cls.iterdir():
            if f.is_file():
                dst = dest_dir / f.name
                try:
                    os.link(f, dst)
                except Exception:
                    # fallback to copy if hardlink fails
                    shutil.copy2(f, dst)
print('Done')
