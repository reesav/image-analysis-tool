"""Measurement functions using YOLO11 segmentation and OpenCV Haar Cascade face detection."""
import cv2
import numpy as np

_YOLO = None
_CASCADE = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO("yolo11s-seg.pt")
    return _YOLO


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(cascade_path)
    return _CASCADE


def measure_people(img_bgr, conf=0.25):
    """Count people and compute % of image covered by them (overlaps counted once)."""
    h, w = img_bgr.shape[:2]
    res = _get_yolo().predict(img_bgr, classes=[0], conf=conf, verbose=False)[0]
    mask = np.zeros((h, w), np.uint8)
    n = 0
    if res.masks is not None:
        for poly in res.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                n += 1
    pct = 100.0 * mask.sum() / (h * w)
    return {"n_people": n, "pct_people": round(pct, 2)}, mask


def measure_faces(img_bgr):
    """Detect faces; return count and size of largest face as % of image area."""
    h, w = img_bgr.shape[:2]
    cascade = _get_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxes = []
    if len(detected) > 0:
        for (x, y, bw, bh) in detected:
            boxes.append((int(x), int(y), int(bw), int(bh)))

    largest = max((bw * bh for _, _, bw, bh in boxes), default=0)
    return {
        "face_detected": "yes" if len(boxes) > 0 else "no",
        "n_faces": len(boxes),
        "largest_face_pct": round(100.0 * largest / (h * w), 2),
    }, boxes


def draw_overlay(img_bgr, mask, boxes):
    over = img_bgr.copy()
    over[mask > 0] = (0.55 * over[mask > 0] + 0.45 * np.array([255, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 0), 3)
    for x, y, bw, bh in boxes:
        cv2.rectangle(over, (x, y), (x + bw, y + bh), (255, 0, 255), 3)
    return over


def measure_image(img_bgr):
    people, mask = measure_people(img_bgr)
    faces, boxes = measure_faces(img_bgr)
    return {**people, **faces}, draw_overlay(img_bgr, mask, boxes)
