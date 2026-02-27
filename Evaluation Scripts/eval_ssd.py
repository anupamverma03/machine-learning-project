import cv2
import os
import time
import csv
from tqdm import tqdm

# ================= PATH CONFIG =================
IMAGE_DIR = "WIDER_subset_big_easy"
GT_FILE = "WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt"

PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.9

SUMMARY_CSV = "resnet_ssd_summary.csv"
DETAIL_CSV = "resnet_ssd_per_image_metrics.csv"

# ================= LOAD MODEL =================
print("Loading ResNet SSD model...")
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model loaded.")

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

# ================= LOAD GT =================
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

for img_path in tqdm(subset_images, desc="Evaluating ResNet SSD"):

    filename = os.path.basename(img_path)

    if filename not in gt_data:
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    (h, w) = img.shape[:2]

    # Create blob from full image
    blob = cv2.dnn.blobFromImage(
        img,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    detected_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            detected_boxes.append((x1, y1, x2 - x1, y2 - y1))

    gt_boxes = gt_data[filename]
    matched_gt = set()

    image_TP = 0
    image_FP = 0

    for det in detected_boxes:
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