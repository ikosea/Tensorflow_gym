import tensorflow as tf
import os

DATA_DIR = "data/cats_vs_dogs"

bad_images = []

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            try:
                img = tf.io.read_file(path)
                tf.image.decode_image(img)
            except Exception:
                bad_images.append(path)
                print("❌ BAD IMAGE:", path)
                os.remove(path)

print("\n✅ DONE")
print("Bad images removed:", len(bad_images))
