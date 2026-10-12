"""Measurement functions using YOLO11 segmentation (people) and YOLO11 pose (faces)."""
import cv2

# Load OpenCV Haar cascade for fallback face detection (frontal face)
_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def _detect_faces_fallback(img_bgr):
    """Return list of face boxes (x, y, w, h) using Haar cascade.
    The cascade works on grayscale images.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detections = _haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]
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


def _deduplicate_boxes(boxes, thresh=0.35):
    """Remove duplicate overlapping boxes for the same face/person."""
    if not boxes:
        return []
    # Sort boxes by area descending
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        x, y, w, h = b
        area = w * h
        dup = False
        for kx, ky, kw, kh in kept:
            ix1 = max(x, kx)
            iy1 = max(y, ky)
            ix2 = min(x + w, kx + kw)
            iy2 = min(y + h, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = area + (kw * kh) - inter
            iou = inter / union if union > 0 else 0
            ios = inter / min(area, kw * kh) if min(area, kw * kh) > 0 else 0
            if iou > thresh or ios > 0.5:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def measure_people(img_bgr, conf=0.25):
    """Count people and compute % of image covered by them (overlaps counted once)."""
    h, w = img_bgr.shape[:2]
    res = _get_yolo_seg().predict(img_bgr, classes=[0], conf=conf, iou=0.45, verbose=False)[0]
    mask = np.zeros((h, w), np.uint8)
    n = 0
    if res.masks is not None:
        for poly in res.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                n += 1
    pct = 100.0 * mask.sum() / (h * w)
    return {"n_people": n, "pct_people": round(pct, 2)}, mask


def measure_faces(
    img_bgr,
    ear_span_multiplier=1.08,
    eye_span_multiplier=2.0,
    dedup_iou_threshold=0.35,
    keypoint_conf_threshold=0.3,
):
    """Detect faces with snug bounding boxes and more accurate area estimation.
    Parameters can be tuned via the UI sliders.
    """
    h, w = img_bgr.shape[:2]
    raw_entries = []  # (box, area)
    # Pose based detection
    try:
        res = _get_yolo_pose().predict(img_bgr, verbose=False)[0]
        if res.keypoints is not None and res.keypoints.xy is not None:
            xy_data = res.keypoints.xy.cpu().numpy()
            conf_data = (
                res.keypoints.conf.cpu().numpy()
                if res.keypoints.conf is not None
                else None
            )
            for idx, kpts in enumerate(xy_data):
                cf = conf_data[idx] if conf_data is not None else [1.0] * len(kpts)
                if cf[0] <= keypoint_conf_threshold:
                    continue
                nose = kpts[0]
                eye_pts = [kpts[i] for i in [1, 2] if cf[i] > keypoint_conf_threshold]
                ear_pts = [kpts[i] for i in [3, 4] if cf[i] > keypoint_conf_threshold]
                avg_eye_y = np.mean([p[1] for p in eye_pts]) if eye_pts else nose[1] - 30.0
                eye_nose_d = max(15.0, abs(nose[1] - avg_eye_y))
                if len(ear_pts) == 2:
                    ear_span = abs(ear_pts[1][0] - ear_pts[0][0])
                    face_w = ear_span * ear_span_multiplier
                    center_x = (ear_pts[0][0] + ear_pts[1][0]) / 2.0
                elif len(eye_pts) == 2:
                    eye_span = abs(eye_pts[1][0] - eye_pts[0][0])
                    face_w = eye_span * eye_span_multiplier
                    center_x = (eye_pts[0][0] + eye_pts[1][0]) / 2.0
                else:
                    face_w = eye_nose_d * 3.0
                    center_x = nose[0]
                top_y = avg_eye_y - 2.2 * eye_nose_d
                bottom_y = nose[1] + 1.6 * eye_nose_d
                bx = max(0, int(center_x - face_w / 2.0))
                by = max(0, int(top_y))
                bw = min(w - bx, int(face_w))
                bh = min(h - by, int(bottom_y - top_y))
                box = (bx, by, bw, bh)
                area = bw * bh * 0.7854
                raw_entries.append((box, area))
    except Exception:
        pass
    # Fallback Haar cascade if no pose boxes
    if not raw_entries:
        for box in _detect_faces_fallback(img_bgr):
            area = box[2] * box[3] * 0.7854
            raw_entries.append((box, area))
    # Deduplicate based on boxes
    boxes_only = [b for b, _ in raw_entries]
    deduped_boxes = _deduplicate_boxes(boxes_only, thresh=dedup_iou_threshold)
    final_entries = [(b, a) for (b, a) in raw_entries if b in deduped_boxes]
    # Add head‑only boxes from person segmentation when missing a face
    try:
        res_seg = _get_yolo_seg().predict(img_bgr, classes=[0], conf=0.25, verbose=False)[0]
        if res_seg.boxes is not None:
            for pb in res_seg.boxes.xyxy.cpu().numpy():
                px1, py1, px2, py2 = pb
                pw, ph = px2 - px1, py2 - py1
                has_face = False
                for fx, fy, fw, fh in deduped_boxes:
                    fc_x = fx + fw / 2.0
                    fc_y = fy + fh / 2.0
                    if px1 <= fc_x <= px2 and py1 <= fc_y <= (py1 + ph * 0.70):
                        has_face = True
                        break
                if not has_face and pw > 30 and ph > 40:
                    fw = int(pw * 0.48)
                    fh = int(ph * 0.38)
                    fx = int(px1 + (pw - fw) * 0.45)
                    fy = int(py1 + ph * 0.05)
                    box = (max(0, fx), max(0, fy), min(w - fx, fw), min(h - fy, fh))
                    area = box[2] * box[3] * 0.7854
                    final_entries.append((box, area))
    except Exception:
        pass
    final_boxes = _deduplicate_boxes([b for b, _ in final_entries], thresh=dedup_iou_threshold)
    largest_area = 0.0
    for b, a in final_entries:
        if b in final_boxes and a > largest_area:
            largest_area = a
    result = {
        "face_detected": "yes" if final_boxes else "no",
        "n_faces": len(final_boxes),
        "largest_face_pct": round(100.0 * largest_area / (h * w), 2),
    }
    return result, final_boxes


def draw_overlay(img_bgr, mask, boxes):
    over = img_bgr.copy()
    over[mask > 0] = (0.55 * over[mask > 0] + 0.45 * np.array([255, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 0), 3)
    for x, y, bw, bh in boxes:
        cv2.rectangle(over, (x, y), (x + bw, y + bh), (255, 0, 255), 3)
    return over


def measure_image(
    img_bgr,
    ear_span_multiplier=1.08,
    eye_span_multiplier=2.0,
    dedup_iou_threshold=0.35,
    keypoint_conf_threshold=0.3,
):
    """Run both people and face measurements.
    The face measurement parameters are exposed as UI sliders.
    """
    people, mask = measure_people(img_bgr)
    faces, boxes = measure_faces(
        img_bgr,
        ear_span_multiplier=ear_span_multiplier,
        eye_span_multiplier=eye_span_multiplier,
        dedup_iou_threshold=dedup_iou_threshold,
        keypoint_conf_threshold=keypoint_conf_threshold,
    )
    return {**people, **faces}, draw_overlay(img_bgr, mask, boxes)
