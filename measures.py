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
    """Detect faces; ensures each person has exactly ONE face box without duplicates."""
    h, w = img_bgr.shape[:2]
    raw_boxes = []

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
                    if cf > 0.25 and (pt[0] > 0 or pt[1] > 0):
                        face_pts.append((kidx, pt[0], pt[1]))

                if face_pts:
                    pts_arr = np.array([[x, y] for _, x, y in face_pts])
                    min_x, min_y = np.min(pts_arr, axis=0)
                    max_x, max_y = np.max(pts_arr, axis=0)
                    center_x = float(np.mean(pts_arr[:, 0]))
                    center_y = float(np.mean(pts_arr[:, 1]))

                    kdict = {k: (x, y) for k, x, y in face_pts}
                    if 3 in kdict and 4 in kdict:
                        span_w = abs(kdict[4][0] - kdict[3][0])
                        face_w = max(span_w * 1.35, 30.0)
                    elif 1 in kdict and 2 in kdict:
                        span_w = abs(kdict[2][0] - kdict[1][0])
                        face_w = max(span_w * 2.5, 30.0)
                    else:
                        face_w = max((max_x - min_x) * 2.2, 35.0)

                    face_h = face_w * 1.35
                    if 0 in kdict:
                        center_y = kdict[0][1] - face_h * 0.1

                    bx = max(0, int(center_x - face_w * 0.5))
                    by = max(0, int(center_y - face_h * 0.52))
                    bw = min(w - bx, int(face_w))
                    bh = min(h - by, int(face_h))
                    raw_boxes.append((bx, by, bw, bh))
    except Exception:
        raw_boxes = []

    # 1. Deduplicate any overlapping face boxes (eliminates multiple boxes on the same face)
    deduped = _deduplicate_boxes(raw_boxes, thresh=0.35)

    # 2. Check segmentation detections to ensure any detected figure without pose also gets a face box
    try:
        res_seg = _get_yolo_seg().predict(img_bgr, classes=[0], conf=0.25, verbose=False)[0]
        if res_seg.boxes is not None:
            for pb in res_seg.boxes.xyxy.cpu().numpy():
                px1, py1, px2, py2 = pb
                pw, ph = px2 - px1, py2 - py1
                # Check if this person already has a face box in their upper region
                has_face = False
                for fx, fy, fw, fh in deduped:
                    fc_x = fx + fw / 2.0
                    fc_y = fy + fh / 2.0
                    if px1 <= fc_x <= px2 and py1 <= fc_y <= (py1 + ph * 0.65):
                        has_face = True
                        break
                if not has_face and pw > 30 and ph > 40:
                    fw = int(pw * 0.55)
                    fh = int(ph * 0.35)
                    fx = int(px1 + (pw - fw) / 2.0)
                    fy = int(py1)
                    deduped.append((max(0, fx), max(0, fy), min(w - fx, fw), min(h - fy, fh)))
    except Exception:
        pass

    final_boxes = _deduplicate_boxes(deduped, thresh=0.35)
    largest = max((bw * bh for _, _, bw, bh in final_boxes), default=0)

    return {
        "face_detected": "yes" if len(final_boxes) > 0 else "no",
        "n_faces": len(final_boxes),
        "largest_face_pct": round(100.0 * largest / (h * w), 2),
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
