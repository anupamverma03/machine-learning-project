import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

# ================= CONFIG =================
HAAR_PATH = "haarcascade_frontalface_default.xml"
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

WIDER_TXT = "WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt"
IMAGE_DIR = "WIDER_subset_big_easy"

IOU_THRESHOLD = 0.5
THRESHOLDS = np.arange(0.3, 0.95, 0.05)
N_JOBS = -1  # Use all CPU cores

SCALE_FACTOR = 1.05 #optimal val = 1.05
MIN_NEIGHBORS = 3
MIN_SIZE = (30, 30)

# ================= IOU =================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0


# ================= Parse WIDERFACE dataset =================
def parse_wider():
    annotations = {}
    with open(WIDER_TXT, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        img_path = lines[i].strip()
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        boxes = []
        for _ in range(num_faces):
            parts = list(map(int, lines[i].strip().split()))
            x, y, w, h = parts[:4]
            boxes.append([x, y, x+w, y+h])
            i += 1

        img_name = os.path.basename(img_path)
        if os.path.exists(os.path.join(IMAGE_DIR, img_name)):
            annotations[img_name] = boxes

    return annotations


# ================= Detection Worker =================
def detect_image(img_name):
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        return img_name, []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE
    )

    detections = []

    for (fx, fy, fw, fh) in faces:
        roi = (
            max(0, fx - 20),
            max(0, fy - 20),
            min(image.shape[1], fw + 40),
            min(image.shape[0], fh + 40)
        )

        face_crop = image[
            roi[1]:roi[1]+roi[3],
            roi[0]:roi[0]+roi[2]
        ]

        if face_crop.size > 0:
            blob = cv2.dnn.blobFromImage(
                face_crop,
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )
            net.setInput(blob)
            output = net.forward()

            for i in range(output.shape[2]):
                conf = float(output[0, 0, i, 2])
                box = output[0, 0, i, 3:7]

                bx1 = int(box[0]*roi[2]) + roi[0]
                by1 = int(box[1]*roi[3]) + roi[1]
                bx2 = int(box[2]*roi[2]) + roi[0]
                by2 = int(box[3]*roi[3]) + roi[1]

                detections.append({
                    "box": [bx1, by1, bx2, by2],
                    "confidence": conf
                })

    return img_name, detections


# ================= MAIN =================
annotations = parse_wider()

print("\nRunning parallel detection...\n")

results = Parallel(n_jobs=N_JOBS)(
    delayed(detect_image)(img)
    for img in tqdm(annotations.keys())
)

all_predictions = dict(results)

# Save cache
with open("detections.pkl", "wb") as f:
    pickle.dump(all_predictions, f)

# ================= PR CURVE COMPUTATION =================
pr_data = []

for threshold in THRESHOLDS:
    TP = FP = FN = 0

    for img_name, gt_boxes in annotations.items():

        preds = [
            d["box"]
            for d in all_predictions[img_name]
            if d["confidence"] > threshold
        ]

        matched = []

        for det in preds:
            match_flag = False
            for gt in gt_boxes:
                if compute_iou(det, gt) > IOU_THRESHOLD:
                    TP += 1
                    matched.append(gt)
                    match_flag = True
                    break
            if not match_flag:
                FP += 1

        for gt in gt_boxes:
            if gt not in matched:
                FN += 1

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    pr_data.append([threshold, precision, recall])

df = pd.DataFrame(pr_data, columns=["threshold","precision","recall"])
df.to_csv("pr_curve.csv", index=False)

df_sorted = df.sort_values("recall")
mAP = np.trapz(df_sorted["precision"], df_sorted["recall"])
# ================= PLOT CURVE =================
plt.figure()
plt.plot(df["recall"], df["precision"], marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PR Curve")
plt.grid(True)
plt.savefig("pr_curve.png", dpi=300)
plt.close()
# ================= RESULTS =================
print("\n✔ PR curve saved")
print(f"Curr papameters: IOU={IOU_THRESHOLD}, SCALE FACTOR={SCALE_FACTOR}, NEIGHBORS={MIN_NEIGHBORS}, MIN_SIZE={MIN_SIZE}")
print("✔ mAP:", round(mAP,3))