import os, shutil, random

src = 'dataset'
train_dir = os.path.join(src, 'train')
val_dir = os.path.join(src, 'val')

classes = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d not in ['train', 'val']]
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in classes:
    cls_path = os.path.join(src, cls)
    imgs = os.listdir(cls_path)
    random.shuffle(imgs)
    split = int(0.8 * len(imgs))
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    for img in imgs[:split]:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in imgs[split:]:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
