import os
import shutil

# ==============================
# CONFIGURATION
# ==============================

ANNOTATION_FILE = "WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt"
IMAGE_ROOT = "WIDERFACE/WIDER_val/images"
OUTPUT_DIR = "WIDER_subset_big_easy"
MIN_FACE_SIZE = 80          # Minimum width & height
MAX_FACES = 2               # 1–2 faces only
MIN_FACES = 1

# ==============================
# CREATE OUTPUT DIRECTORY
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_big_face(face):
    """
    face format:
    x y w h blur expression illumination invalid occlusion pose
    """
    x, y, w, h = face[:4]
    return w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE

def parse_wider_annotations():
    selected_images = []

    with open(ANNOTATION_FILE, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        faces = []
        for _ in range(num_faces):
            face_data = list(map(int, lines[i].strip().split()))
            faces.append(face_data)
            i += 1

        # Filter 1–2 faces
        if not (MIN_FACES <= num_faces <= MAX_FACES):
            continue

        # Keep only big faces
        big_faces = [face for face in faces if is_big_face(face)]

        if len(big_faces) == num_faces:
            selected_images.append(image_path)

    return selected_images

def copy_selected_images(image_list):
    for img_rel_path in image_list:
        src_path = os.path.join(IMAGE_ROOT, img_rel_path)
        dst_path = os.path.join(OUTPUT_DIR, os.path.basename(img_rel_path))

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    print(f"Copied {len(image_list)} images to {OUTPUT_DIR}")

if __name__ == "__main__":
    selected = parse_wider_annotations()
    copy_selected_images(selected)
