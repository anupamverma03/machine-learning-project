import cv2
import os
import time
import csv
from tqdm import tqdm

# ================= PATH CONFIG =================
IMAGE_DIR = "WIDER_subset_big_easy"
GT_FILE = "WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

IOU_THRESHOLD = 0.5
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 3
MIN_SIZE = (30, 30)

SUMMARY_CSV = "evaluation_summary.csv"
DETAIL_CSV = "per_image_metrics.csv"

# ================= LOAD HAAR =================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("Error loading Haar cascade!")
    exit()

# ================= IoU FUNCTION =================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union = areaA + areaB - inter_area
    if union == 0:
        return 0

    return inter_area / union

# ================= LOAD GT (MATCH BY FILENAME) =================
def load_wider_gt_by_filename(gt_path):
    gt_data = {}

    with open(gt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        filename = os.path.basename(image_path)
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        boxes = []
        for _ in range(num_faces):
            parts = lines[i].strip().split()
            x, y, w, h = map(int, parts[:4])
            boxes.append((x, y, w, h))
            i += 1

        gt_data[filename] = boxes

    return gt_data

print("Loading Ground Truth...")
gt_data = load_wider_gt_by_filename(GT_FILE)
print("GT Loaded.")

# ================= GET SUBSET IMAGES =================
subset_images = []

for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            subset_images.append(os.path.join(root, file))

print(f"Total subset images found: {len(subset_images)}")

# ================= EVALUATION =================
TP = 0
FP = 0
FN = 0

per_image_results = []

start_time = time.time()

for img_path in tqdm(subset_images, desc="Evaluating"):

    filename = os.path.basename(img_path)

    if filename not in gt_data:
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE
    )

    gt_boxes = gt_data[filename]
    matched_gt = set()

    image_TP = 0
    image_FP = 0

    for det in detections:
        match_found = False

        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue

            iou = compute_iou(det, gt)

            if iou >= IOU_THRESHOLD:
                image_TP += 1
                matched_gt.add(idx)
                match_found = True
                break

        if not match_found:
            image_FP += 1

    image_FN = len(gt_boxes) - len(matched_gt)

    TP += image_TP
    FP += image_FP
    FN += image_FN

    per_image_results.append([
        filename,
        image_TP,
        image_FP,
        image_FN
    ])

end_time = time.time()

# ================= METRICS =================
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

total_time = end_time - start_time
fps = len(subset_images) / total_time if total_time > 0 else 0

# ================= SAVE SUMMARY =================
with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Total TP", "Total FP", "Total FN",
                     "Precision", "Recall", "F1 Score", "Evaluation FPS"])
    writer.writerow([TP, FP, FN, precision, recall, f1_score, fps])

# ================= SAVE PER IMAGE =================
with open(DETAIL_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "TP", "FP", "FN"])
    writer.writerows(per_image_results)

# ================= PRINT RESULTS =================
print("\n========== FINAL RESULTS ==========")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print("-----------------------------------")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("-----------------------------------")
print(f"Evaluation FPS: {fps:.2f}")
print("Saved:")
print(" -", SUMMARY_CSV)
print(" -", DETAIL_CSV)
print("===================================\n")