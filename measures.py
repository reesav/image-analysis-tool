"""Measurement functions. Each returns a dict of values (and optionally drawing data).
To add a new measure later (brightness, saturation...), write another function
and call it from measure_image()."""
import cv2
import numpy as np

_YOLO = None
_FACE = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO("yolo11s-seg.pt")  # downloads on first run
    return _YOLO


def _get_face():
    global _FACE
    if _FACE is None:
        import mediapipe as mp
        _FACE = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
    return _FACE


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
    out = _get_face().process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    boxes = []
    for d in (out.detections or []):
        b = d.location_data.relative_bounding_box
        x, y = max(0, int(b.xmin * w)), max(0, int(b.ymin * h))
        bw, bh = int(b.width * w), int(b.height * h)
        boxes.append((x, y, bw, bh))
    largest = max((bw * bh for _, _, bw, bh in boxes), default=0)
    return {
        "face_detected": "yes" if boxes else "no",
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
