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
    """Detect faces using YOLO pose keypoints; returns full face boxes (forehead to chin, cheek to cheek)."""
    h, w = img_bgr.shape[:2]
    boxes = []

    try:
        res = _get_yolo_pose().predict(img_bgr, verbose=False)[0]
        if res.keypoints is not None and res.keypoints.xy is not None:
            xy_data = res.keypoints.xy.cpu().numpy()
            conf_data = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
            person_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None

            for idx, person_kpts in enumerate(xy_data):
                # Keypoints: 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear, 5: left_sh, 6: right_sh
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

                    # Find ear span and eye span if available
                    kdict = {k: (x, y) for k, x, y in face_pts}
                    has_ears = (3 in kdict and 4 in kdict)
                    has_eyes = (1 in kdict and 2 in kdict)

                    if has_ears:
                        span_w = abs(kdict[4][0] - kdict[3][0])
                        face_w = max(span_w * 1.35, 30.0)
                    elif has_eyes:
                        span_w = abs(kdict[2][0] - kdict[1][0])
                        face_w = max(span_w * 2.5, 30.0)
                    else:
                        face_w = max((max_x - min_x) * 2.2, 35.0)

                    # Estimate full face height (from forehead/hair to chin)
                    face_h = face_w * 1.35

                    # If nose is detected, shift center slightly above nose towards eyes
                    if 0 in kdict:
                        center_y = kdict[0][1] - face_h * 0.1

                    # If shoulders are detected, use shoulders to anchor chin
                    sh_y = []
                    if conf_data is not None and len(person_kpts) > 6:
                        if conf_data[idx][5] > 0.3 and person_kpts[5][1] > 0:
                            sh_y.append(person_kpts[5][1])
                        if conf_data[idx][6] > 0.3 and person_kpts[6][1] > 0:
                            sh_y.append(person_kpts[6][1])

                    bx = max(0, int(center_x - face_w * 0.5))
                    by = max(0, int(center_y - face_h * 0.52))
                    bw = min(w - bx, int(face_w))
                    bh = min(h - by, int(face_h))

                    if sh_y:
                        avg_sh = float(np.mean(sh_y))
                        if by + bh > avg_sh:
                            bh = max(20, int(avg_sh - by))

                    boxes.append((bx, by, bw, bh))

                elif person_boxes is not None and idx < len(person_boxes):
                    # Fallback to top portion of person box if face keypoints were obscured
                    px1, py1, px2, py2 = person_boxes[idx]
                    pw, ph = px2 - px1, py2 - py1
                    if pw > 15 and ph > 30:
                        fw = int(pw * 0.5)
                        fh = int(ph * 0.22)
                        fx = int(px1 + (pw - fw) / 2)
                        fy = int(py1)
                        boxes.append((max(0, fx), max(0, fy), min(w - fx, fw), min(h - fy, fh)))

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
