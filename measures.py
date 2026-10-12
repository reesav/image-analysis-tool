"""Measurement functions using YOLO11 segmentation (people) and YOLO11 pose (faces)."""
import cv2
import numpy as np

_YOLO_SEG = None
_YOLO_POSE = None


def _get_yolo_seg():
    global _YOLO_SEG
    if _YOLO_SEG is None:
        from ultralytics import YOLO
        _YOLO_SEG = YOLO("yolo11s-seg.pt")
    return _YOLO_SEG


def _get_yolo_pose():
    global _YOLO_POSE
    if _YOLO_POSE is None:
        from ultralytics import YOLO
        _YOLO_POSE = YOLO("yolo11n-pose.pt")
    return _YOLO_POSE


def measure_people(img_bgr, conf=0.25):
    """Count people and compute % of image covered by them (overlaps counted once)."""
    h, w = img_bgr.shape[:2]
    res = _get_yolo_seg().predict(img_bgr, classes=[0], conf=conf, verbose=False)[0]
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
    """Detect faces using YOLO pose keypoints (nose, eyes, ears); return count and largest face area."""
    h, w = img_bgr.shape[:2]
    boxes = []

    try:
        res = _get_yolo_pose().predict(img_bgr, verbose=False)[0]
        if res.keypoints is not None and res.keypoints.xy is not None:
            xy_data = res.keypoints.xy.cpu().numpy()
            conf_data = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
            for idx, person_kpts in enumerate(xy_data):
                face_pts = []
                for kidx in range(min(5, len(person_kpts))):
                    pt = person_kpts[kidx]
                    cf = conf_data[idx][kidx] if conf_data is not None else 1.0
                    if cf > 0.3 and (pt[0] > 0 or pt[1] > 0):
                        face_pts.append(pt)
                if face_pts:
                    pts = np.array(face_pts)
                    xmin, ymin = np.min(pts, axis=0)
                    xmax, ymax = np.max(pts, axis=0)
                    pad_w = max(10, int((xmax - xmin) * 0.3))
                    pad_h = max(10, int((ymax - ymin) * 0.4))
                    bx = max(0, int(xmin - pad_w))
                    by = max(0, int(ymin - pad_h))
                    bw = min(w - bx, int(xmax - xmin + 2 * pad_w))
                    bh = min(h - by, int(ymax - ymin + 2 * pad_h))
                    boxes.append((bx, by, bw, bh))
    except Exception:
        boxes = []

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
