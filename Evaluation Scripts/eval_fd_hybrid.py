import cv2
import os
import time
import csv
import numpy as np

# ================= CONFIG =================
HAAR_PATH = "haarcascade_frontalface_default.xml"
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

WIDER_TXT = "WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt"
IMAGE_DIR = "WIDER_subset_big_easy"

CSV_OUTPUT = "hybrid_wider_results_corrected.csv"
SUMMARY_OUTPUT = "hybrid_wider_summary_corrected.csv"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.9

SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 3
MIN_SIZE = (30, 30)
# ===========================================

face_cascade = cv2.CascadeClassifier(HAAR_PATH)

net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


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


# ================= NMS =================
def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        suppress = [len(idxs)-1]

        for pos in range(len(idxs)-1):
            i = idxs[pos]

            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            overlap = (w*h) / areas[i]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick].astype("int").tolist()


# ================= Parse WIDER =================
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


# ================= Corrected Hybrid Detection =================                            
def hybrid_detect(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start = time.time()

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE
    )

    detections = []

    for (fx, fy, fw, fh) in faces:

        # ROI calculation
        x1 = max(0, fx - 20)
        y1 = max(0, fy - 20)
        x2 = min(image.shape[1], fx + fw + 20)
        y2 = min(image.shape[0], fy + fh + 20)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(
            face_crop,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        output = net.forward()

        for i in range(output.shape[2]):
            conf = output[0, 0, i, 2]

            if conf > CONF_THRESHOLD:
                box = output[0, 0, i, 3:7]

                bx1 = int(box[0]*(x2-x1)) + x1
                by1 = int(box[1]*(y2-y1)) + y1
                bx2 = int(box[2]*(x2-x1)) + x1
                by2 = int(box[3]*(y2-y1)) + y1

                detections.append([bx1, by1, bx2, by2])

    # Apply NMS
    detections = non_max_suppression(detections)

    inference_ms = (time.time() - start) * 1000

    return detections, inference_ms


# ================= MAIN EVALUATION =================
annotations = parse_wider()

total_images = len(annotations)
total_TP = total_FP = total_FN = 0
times = []

start_total = time.time()

print("\nStarting Hybrid Evaluation...")
print(f"Total Images: {total_images}")
print("====================================\n")

with open(CSV_OUTPUT, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image","TP","FP","FN","precision","recall","f1","inference_ms"])

    for idx, (img_name, gt_boxes) in enumerate(annotations.items(), 1):

        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        detections, inf_time = hybrid_detect(image)

        TP = FP = FN = 0
        matched_gt = []

        for det in detections:
            matched_flag = False
            for gt in gt_boxes:
                if compute_iou(det, gt) > IOU_THRESHOLD:
                    TP += 1
                    matched_gt.append(gt)
                    matched_flag = True
                    break
            if not matched_flag:
                FP += 1

        for gt in gt_boxes:
            if gt not in matched_gt:
                FN += 1

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        total_TP += TP
        total_FP += FP
        total_FN += FN
        times.append(inf_time)

        writer.writerow([
            img_name,
            TP, FP, FN,
            round(precision,3),
            round(recall,3),
            round(f1,3),
            round(inf_time,2)
        ])

        # ===== LIVE STATUS =====
        elapsed = time.time() - start_total
        avg_time = np.mean(times)
        est_total_time = avg_time * total_images / 1000
        remaining = max(0, est_total_time - elapsed)

        print(
            f"[{idx}/{total_images}] "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {remaining:.1f}s | "
            f"Current F1: {f1:.3f}",
            end="\r"
        )

# ================= FINAL SUMMARY =================
precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

avg_time = np.mean(times)
fps = 1000 / avg_time if avg_time > 0 else 0
total_time = time.time() - start_total

# -------- Save Summary CSV --------
with open(SUMMARY_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Total Images",
        "Total TP",
        "Total FP",
        "Total FN",
        "Precision",
        "Recall",
        "F1 Score",
        "Avg Inference (ms)",
        "Evaluation FPS",
        "Total Eval Time (sec)",
        "IOU Threshold",
        "Confidence Threshold",
        "Scale Factor",
        "Min Neighbors",
        "Min Size"
    ])

    writer.writerow([
        total_images,
        total_TP,
        total_FP,
        total_FN,
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(avg_time, 2),
        round(fps, 2),
        round(total_time, 2),
        IOU_THRESHOLD,
        CONF_THRESHOLD,
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        MIN_SIZE
    ])

# -------- Print Results --------
print("\n\n====================================")
print("HYBRID HAAR + RES10 SSD (FINAL)")
print("------------------------------------")
print(f"Total Images: {total_images}")
print(f"TP: {total_TP}")
print(f"FP: {total_FP}")
print(f"FN: {total_FN}")
print("------------------------------------")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("------------------------------------")
print(f"Avg Inference Time: {avg_time:.2f} ms")
print(f"Evaluation FPS: {fps:.2f}")
print(f"Total Eval Time: {total_time:.2f} sec")
print("====================================")
print("Saved:")
print(" -", CSV_OUTPUT)
print(" -", SUMMARY_OUTPUT)