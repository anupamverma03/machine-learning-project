import os

GT_FILE = r"c:\datasets\WIDERFACE\WIDER_val\wider_face_val_bbx_gt.txt"
OUT_DIR = r"c:\datasets\WIDERFACE\WIDER_val"

easy, medium, hard = set(), set(), set()

with open(GT_FILE, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    img_path = lines[i].strip()
    num_faces = int(lines[i + 1].strip())
    i += 2

    max_area = 0
    for _ in range(num_faces):
        x, y, w, h, *_ = map(int, lines[i].split())
        max_area = max(max_area, w * h)
        i += 1

    if max_area >= 64 * 64:
        easy.add(img_path)
    elif max_area >= 32 * 32:
        medium.add(img_path)
    else:
        hard.add(img_path)

# Save split files
def save(name, data):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        for item in sorted(data):
            f.write(item + "\n")
    print(f"Saved {len(data)} images → {name}")

save("wider_easy_val.txt", easy)
save("wider_medium_val.txt", medium)
save("wider_hard_val.txt", hard)
