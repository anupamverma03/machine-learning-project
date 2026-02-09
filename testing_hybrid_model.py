import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

# ===================== PATHS =====================
DATASET_ROOT = r"C:\datasets\WIDERFACE\WIDER_val"
IMG_ROOT = os.path.join(DATASET_ROOT, "images")
GT_FILE = os.path.join(DATASET_ROOT, "wider_face_val_bbx_gt.txt")

SPLITS = {
    "easy": os.path.join(DATASET_ROOT, "wider_easy_val.txt"),
    "medium": os.path.join(DATASET_ROOT, "wider_medium_val.txt"),
    "hard": os.path.join(DATASET_ROOT, "wider_hard_val.txt"),
}

MODEL_PATH = r"C:\models\face_detector.onnx"
OUTPUT_DIR = r"C:\evaluation\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== PARAMETERS =====================
CONF_THRESH = 0.25
IOU_THRESH = 0.3
INPUT_SIZE = (320, 320)
MIN_FACE_SIZE = 32   # <<< THIS FIXES ZERO TP PROBLEM

# ===================== LOAD MODEL =====================
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ===================== IOU FUNCTION =====================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ===================== LOAD & FILTER GT =====================
gt = {}

with open(GT_FILE, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    img_name = lines[i].strip()
    num_faces = int(lines[i + 1])
    i += 2

    boxes = []
    for _ in range(num_faces):
        vals = list(map(int, lines[i].split()))
        i += 1

        x, y, w, h = vals[0:4]
        invalid = vals[7]

        # ---- CRITICAL FILTERING ----
        if invalid == 0 and w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            boxes.append((x, y, x + w, y + h))

    gt[img_name] = boxes

# ===================== LOAD SPLITS =====================
split_imgs = {}
for s, p in SPLITS.items():
    with open(p) as f:
        split_imgs[s] = [l.strip() for l in f.readlines()]

# ===================== METRICS STORAGE =====================
metrics = {s: {"TP": 0, "FP": 0, "FN": 0} for s in SPLITS}
pr_data = {s: {"scores": [], "labels": []} for s in SPLITS}

# ===================== EVALUATION =====================
for split in SPLITS:
    print(f"\nEvaluating {split.upper()} split")

    for img_rel in tqdm(split_imgs[split]):
        img_path = os.path.join(IMG_ROOT, img_rel)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        sx = W / INPUT_SIZE[0]
        sy = H / INPUT_SIZE[1]

        blob = cv2.dnn.blobFromImage(
            img, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False
        )
        net.setInput(blob)
        detections = net.forward()[0]

        preds, scores = [], []

        for det in detections:
            score = float(det[4])
            if score < CONF_THRESH:
                continue

            x, y, w, h = det[0:4]
            x = int(x * sx)
            y = int(y * sy)
            w = int(w * sx)
            h = int(h * sy)

            preds.append((x, y, x + w, y + h))
            scores.append(score)

        gt_boxes = gt.get(img_rel, [])
        matched = set()

        for pbox, sc in zip(preds, scores):
            hit = False
            for gi, gtbox in enumerate(gt_boxes):
                if gi in matched:
                    continue
                if iou(pbox, gtbox) >= IOU_THRESH:
                    hit = True
                    matched.add(gi)
                    metrics[split]["TP"] += 1
                    pr_data[split]["scores"].append(sc)
                    pr_data[split]["labels"].append(1)
                    break

            if not hit:
                metrics[split]["FP"] += 1
                pr_data[split]["scores"].append(sc)
                pr_data[split]["labels"].append(0)

        metrics[split]["FN"] += len(gt_boxes) - len(matched)

# ===================== SAVE CSV =====================
rows = []
for s in metrics:
    TP, FP, FN = metrics[s]["TP"], metrics[s]["FP"], metrics[s]["FN"]
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    rows.append([s, TP, FP, FN, precision, recall, f1])

df = pd.DataFrame(rows, columns=[
    "split", "TP", "FP", "FN", "precision", "recall", "f1_score"
])

csv_path = os.path.join(OUTPUT_DIR, "widerface_results_filtered.csv")
df.to_csv(csv_path, index=False)
print(f"\n✔ Results saved to {csv_path}")

# ===================== PR CURVES =====================
plt.figure(figsize=(7, 6))

for s in pr_data:
    if not pr_data[s]["labels"]:
        continue

    p, r, _ = precision_recall_curve(
        pr_data[s]["labels"], pr_data[s]["scores"]
    )
    ap = average_precision_score(
        pr_data[s]["labels"], pr_data[s]["scores"]
    )
    plt.plot(r, p, label=f"{s} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves – SSD (Filtered WIDER FACE)")
plt.grid(True)
plt.legend()
plt.tight_layout()

pr_path = os.path.join(OUTPUT_DIR, "pr_curve_filtered.png")
plt.savefig(pr_path)
plt.show()

print(f"✔ PR curve saved to {pr_path}")
