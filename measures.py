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


def measure_faces(img_bgr):
    """Detect faces with snug bounding boxes fitting hairline to chin, cheek to cheek."""
    h, w = img_bgr.shape[:2]
    raw_boxes = []

    try:
        res = _get_yolo_pose().predict(img_bgr, verbose=False)[0]
        if res.keypoints is not None and res.keypoints.xy is not None:
            xy_data = res.keypoints.xy.cpu().numpy()
            conf_data = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None

            for idx, kpts in enumerate(xy_data):
                cf = conf_data[idx] if conf_data is not None else [1.0] * len(kpts)
                # Check for nose and eyes/ears
                if cf[0] > 0.25 and (cf[1] > 0.25 or cf[2] > 0.25):
                    nose = kpts[0]
                    eye_pts = [kpts[i] for i in [1, 2] if cf[i] > 0.25 and kpts[i][0] > 0]
                    ear_pts = [kpts[i] for i in [3, 4] if cf[i] > 0.25 and kpts[i][0] > 0]

                    avg_eye_y = np.mean([p[1] for p in eye_pts]) if eye_pts else nose[1] - 30.0
                    eye_nose_d = max(15.0, abs(nose[1] - avg_eye_y))

                    # Snug width (ear to ear or 2x eye span)
                    if len(ear_pts) == 2:
                        ear_span = abs(ear_pts[1][0] - ear_pts[0][0])
                        face_w = ear_span * 1.08
                        center_x = (ear_pts[0][0] + ear_pts[1][0]) / 2.0
                    elif len(eye_pts) == 2:
                        eye_span = abs(eye_pts[1][0] - eye_pts[0][0])
                        face_w = eye_span * 2.0
                        center_x = (eye_pts[0][0] + eye_pts[1][0]) / 2.0
                    else:
                        face_w = eye_nose_d * 3.0
                        center_x = nose[0]

                    # Snug height (forehead hairline to chin)
                    top_y = avg_eye_y - 2.2 * eye_nose_d
                    bottom_y = nose[1] + 1.6 * eye_nose_d

                    bx = max(0, int(center_x - face_w / 2.0))
                    by = max(0, int(top_y))
                    bw = min(w - bx, int(face_w))
                    bh = min(h - by, int(bottom_y - top_y))
                    raw_boxes.append((bx, by, bw, bh))
    except Exception:
        raw_boxes = []

    # 1. Deduplicate any overlapping face boxes
    deduped = _deduplicate_boxes(raw_boxes, thresh=0.35)

    # 2. Add snug head box for detected person segments without pose keypoints (e.g. held masks)
    try:
        res_seg = _get_yolo_seg().predict(img_bgr, classes=[0], conf=0.25, verbose=False)[0]
        if res_seg.boxes is not None:
            for pb in res_seg.boxes.xyxy.cpu().numpy():
                px1, py1, px2, py2 = pb
                pw, ph = px2 - px1, py2 - py1
                has_face = False
                for fx, fy, fw, fh in deduped:
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
                    deduped.append((max(0, fx), max(0, fy), min(w - fx, fw), min(h - fy, fh)))
    except Exception:
        pass

    final_boxes = _deduplicate_boxes(deduped, thresh=0.35)

    # Calculate actual face area using snug box dimensions
    largest = max((bw * bh for _, _, bw, bh in final_boxes), default=0)
    # True oval face area (pi / 4 ≈ 0.7854 of the bounding box)
    largest_oval_area = largest * 0.7854

    return {
        "face_detected": "yes" if len(final_boxes) > 0 else "no",
        "n_faces": len(final_boxes),
        "largest_face_pct": round(100.0 * largest_oval_area / (h * w), 2),
    }, final_boxes


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
