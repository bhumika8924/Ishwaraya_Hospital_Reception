"""
Unique visitor presence counter for the hospital reception camera.

Counts unique non-staff visitors currently present in the public/visitor area.
People inside the configured reception/receptionist zone are excluded from the
visitor count.
"""
import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
DEFAULT_VIDEO_CANDIDATES = [
    os.path.join(BASE_DIR, "vdo2.mp4"),
    os.path.join(BASE_DIR, "vdo5.mp4"),
    os.path.join(BASE_DIR, "Screen Recording 2026-05-13 140905.mp4"),
    os.path.join(BASE_DIR, "five_min_vdo.mp4"),
]
UNIFORM_REFERENCE_PATH = os.path.join(BASE_DIR, "receptionist_uniform_ref.png")
RECEPTION_ZONE_PATH = os.path.join(BASE_DIR, "reception_zone.json")

# Every frame is resized to this size before analysis.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# reception_zone.json was drawn on a 1920x1080 frame in the existing app.
RECEPTION_REF_WIDTH = 1920
RECEPTION_REF_HEIGHT = 1080
DEFAULT_RECEPTION_ZONE = np.array(
    [[500, 110], [1600, 100], [1850, 1080], [520, 1080]], dtype=np.float64
)

# Detection tuning.
YOLO_CONF = 0.35
YOLO_IOU = 0.45
MIN_BOX_HEIGHT = 70
MIN_BOX_WIDTH = 32
MIN_VISITOR_BOX_HEIGHT = 120
MIN_VISITOR_BOX_WIDTH = 45

# Reception zone exclusion. Any person with this much box overlap is not a visitor.
MIN_RECEPTION_ZONE_OVERLAP = 0.20

# Staff filtering.
STAFF_CONFIRM_FRAMES = 3
UNIFORM_MATCH_THRESHOLD = 0.55
MIN_RED_LANYARD_PIXELS = 20
STAFF_REID_SIMILARITY = 0.86

# Unique visitor matching.
VISITOR_REID_SIMILARITY = 0.82
MAX_SIGNATURES_PER_PERSON = 12
PRESENT_GRACE_FRAMES = 15


@dataclass
class PersonRecord:
    person_id: int
    first_track_id: int
    first_frame_num: int
    last_seen_frame: int
    signatures: list
    track_ids: set


def parse_args():
    default_video = next(
        (path for path in DEFAULT_VIDEO_CANDIDATES if os.path.isfile(path)),
        DEFAULT_VIDEO_CANDIDATES[0],
    )
    parser = argparse.ArgumentParser(description="Count unique visitors present.")
    parser.add_argument(
        "--video",
        default=os.getenv("VISITOR_VIDEO_PATH", default_video),
        help="Path to the CCTV video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening an OpenCV preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick testing. 0 means full video.",
    )
    return parser.parse_args()


def make_hs_hist(image, bins=(50, 60)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def load_reference_hist(path):
    ref_img = cv2.imread(path)
    if ref_img is None:
        print(f"Warning: could not read uniform reference: {path}")
        return None
    return make_hs_hist(ref_img)


def as_polygon_config(raw, default_ref_w, default_ref_h):
    if isinstance(raw, dict):
        points = raw.get("points") or raw.get("polygon") or raw.get("zone")
        ref_w = int(raw.get("reference_width", default_ref_w))
        ref_h = int(raw.get("reference_height", default_ref_h))
        if points is None:
            raise ValueError("Polygon JSON dict must contain points/polygon/zone.")
        return np.array(points, dtype=np.float64), ref_w, ref_h
    return np.array(raw, dtype=np.float64), default_ref_w, default_ref_h


def load_polygon(path, default_points, default_ref_w, default_ref_h):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        points, ref_w, ref_h = as_polygon_config(raw, default_ref_w, default_ref_h)
        if points.shape != (4, 2):
            raise ValueError("Polygon must contain exactly four [x, y] points.")
        return points, ref_w, ref_h
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        if not isinstance(exc, FileNotFoundError):
            print(f"Warning: could not load {path}: {exc}. Using default zone.")
        return np.array(default_points, copy=True), default_ref_w, default_ref_h


def scale_points(points, ref_w, ref_h, frame_w, frame_h):
    sx = frame_w / float(ref_w)
    sy = frame_h / float(ref_h)
    scaled = []
    for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        scaled.append([int(round(x * sx)), int(round(y * sy))])
    return np.array(scaled, dtype=np.int32)


def polygon_mask(frame_shape, polygon):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def box_overlap_ratio(box, mask):
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    if box_area <= 0:
        return 0.0
    return cv2.countNonZero(mask[y1:y2, x1:x2]) / float(box_area)


def outfit_match_score(frame, box, reference_hist):
    if reference_hist is None:
        return -1.0

    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0

    top = int(crop.shape[0] * 0.25)
    bottom = int(crop.shape[0] * 0.75)
    torso = crop[top:bottom]
    if torso.size == 0:
        return -1.0

    return cv2.compareHist(reference_hist, make_hs_hist(torso), cv2.HISTCMP_CORREL)


def has_id_card_lanyard(frame, box, min_red_pixels=MIN_RED_LANYARD_PIXELS):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0

    neck_top = int(crop.shape[0] * 0.18)
    neck_bottom = int(crop.shape[0] * 0.50)
    neck_region = crop[neck_top:neck_bottom]
    if neck_region.size == 0:
        return False, 0

    hsv = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([140, 50, 100]), np.array([165, 255, 255]))
    red_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
    red_count = cv2.countNonZero(red_mask)
    return red_count >= min_red_pixels, red_count


def clamp_box(box, frame_shape):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def normalized_hs_hist(image):
    if image.size == 0:
        return None
    return make_hs_hist(image, bins=(32, 32))


def person_signature(frame, box):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    body = crop[int(crop.shape[0] * 0.10): int(crop.shape[0] * 0.90)]
    upper = crop[int(crop.shape[0] * 0.18): int(crop.shape[0] * 0.55)]
    lower = crop[int(crop.shape[0] * 0.45): int(crop.shape[0] * 0.92)]

    full_hist = normalized_hs_hist(body)
    upper_hist = normalized_hs_hist(upper)
    lower_hist = normalized_hs_hist(lower)
    if full_hist is None or upper_hist is None or lower_hist is None:
        return None

    return {"full": full_hist, "upper": upper_hist, "lower": lower_hist}


def compare_signatures(signature_a, signature_b):
    if signature_a is None or signature_b is None:
        return -1.0

    full_score = cv2.compareHist(
        signature_a["full"], signature_b["full"], cv2.HISTCMP_CORREL
    )
    upper_score = cv2.compareHist(
        signature_a["upper"], signature_b["upper"], cv2.HISTCMP_CORREL
    )
    lower_score = cv2.compareHist(
        signature_a["lower"], signature_b["lower"], cv2.HISTCMP_CORREL
    )
    return (0.45 * full_score) + (0.35 * upper_score) + (0.20 * lower_score)


def find_matching_record(signature, records, threshold):
    if signature is None:
        return None, -1.0

    best_record = None
    best_score = -1.0
    for record in records:
        for saved_signature in record.signatures:
            score = compare_signatures(signature, saved_signature)
            if score > best_score:
                best_record = record
                best_score = score

    if best_score >= threshold:
        return best_record, best_score
    return None, best_score


def remember_signature(record, track_id, frame_num, signature):
    record.track_ids.add(track_id)
    record.last_seen_frame = frame_num
    if signature is None:
        return

    if len(record.signatures) < MAX_SIGNATURES_PER_PERSON:
        record.signatures.append(signature)
        return

    record.signatures[frame_num % MAX_SIGNATURES_PER_PERSON] = signature


def draw_polygon_overlay(frame, polygon, color, alpha=0.08):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)


def draw_status(frame, present_count, total_unique, staff_count):
    cv2.rectangle(frame, (0, 0), (560, 100), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Visitors present: {present_count}",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Unique visitors seen: {total_unique} | Staff excluded: {staff_count}",
        (16, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        2,
    )


def main():
    args = parse_args()
    video_path = os.path.normpath(os.path.expanduser(args.video))

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model missing: {MODEL_PATH}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(MODEL_PATH)
    reference_hist = load_reference_hist(UNIFORM_REFERENCE_PATH)
    reception_ref, reception_ref_w, reception_ref_h = load_polygon(
        RECEPTION_ZONE_PATH,
        DEFAULT_RECEPTION_ZONE,
        RECEPTION_REF_WIDTH,
        RECEPTION_REF_HEIGHT,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    reception_zone = scale_points(
        reception_ref, reception_ref_w, reception_ref_h, FRAME_WIDTH, FRAME_HEIGHT
    )
    reception_mask = polygon_mask((FRAME_HEIGHT, FRAME_WIDTH, 3), reception_zone)

    if not args.no_display:
        cv2.namedWindow("Visitor Presence Counter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Visitor Presence Counter", FRAME_WIDTH, FRAME_HEIGHT)

    frame_num = 0
    visitor_records = []
    staff_records = []
    visitor_by_track = {}
    staff_by_track = {}
    staff_confirm_count = {}
    staff_track_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        draw_polygon_overlay(frame, reception_zone, (0, 255, 255), alpha=0.06)

        boxes = results[0].boxes
        if boxes.id is not None:
            for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
                if int(cls) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                box_tuple = (x1, y1, x2, y2)
                box_h = y2 - y1
                box_w = x2 - x1
                if box_h < MIN_BOX_HEIGHT or box_w < MIN_BOX_WIDTH:
                    continue

                signature = person_signature(frame, box_tuple)
                reception_overlap = box_overlap_ratio(box_tuple, reception_mask)
                in_reception_area = reception_overlap >= MIN_RECEPTION_ZONE_OVERLAP

                is_staff = track_id in staff_track_ids
                if not is_staff:
                    matched_staff, _ = find_matching_record(
                        signature, staff_records, STAFF_REID_SIMILARITY
                    )
                    if matched_staff is not None:
                        is_staff = True
                        staff_track_ids.add(track_id)
                        staff_by_track[track_id] = matched_staff

                if not is_staff:
                    uniform_score = outfit_match_score(frame, box_tuple, reference_hist)
                    uniform_match = uniform_score >= UNIFORM_MATCH_THRESHOLD
                    lanyard_found, _ = has_id_card_lanyard(frame, box_tuple)
                    if in_reception_area and (uniform_match or lanyard_found):
                        staff_confirm_count[track_id] = (
                            staff_confirm_count.get(track_id, 0) + 1
                        )
                        if staff_confirm_count[track_id] >= STAFF_CONFIRM_FRAMES:
                            is_staff = True
                            staff_track_ids.add(track_id)
                            staff_record = PersonRecord(
                                person_id=len(staff_records) + 1,
                                first_track_id=track_id,
                                first_frame_num=frame_num,
                                last_seen_frame=frame_num,
                                signatures=[],
                                track_ids=set(),
                            )
                            remember_signature(staff_record, track_id, frame_num, signature)
                            staff_records.append(staff_record)
                            staff_by_track[track_id] = staff_record
                    else:
                        staff_confirm_count[track_id] = max(
                            0, staff_confirm_count.get(track_id, 0) - 1
                        )

                if is_staff:
                    staff_record = staff_by_track.get(track_id)
                    if staff_record is not None:
                        remember_signature(staff_record, track_id, frame_num, signature)
                    color = (0, 255, 0)
                    label = f"Staff ID:{track_id}"
                elif in_reception_area:
                    color = (0, 200, 255)
                    label = f"Reception area ID:{track_id}"
                elif box_h >= MIN_VISITOR_BOX_HEIGHT and box_w >= MIN_VISITOR_BOX_WIDTH:
                    visitor_record = visitor_by_track.get(track_id)
                    if visitor_record is None:
                        visitor_record, score = find_matching_record(
                            signature, visitor_records, VISITOR_REID_SIMILARITY
                        )
                        if visitor_record is None:
                            visitor_record = PersonRecord(
                                person_id=len(visitor_records) + 1,
                                first_track_id=track_id,
                                first_frame_num=frame_num,
                                last_seen_frame=frame_num,
                                signatures=[],
                                track_ids=set(),
                            )
                            visitor_records.append(visitor_record)
                            print(
                                f"[VISITOR] frame={frame_num} visitor "
                                f"#{visitor_record.person_id}, track ID {track_id}"
                            )
                        visitor_by_track[track_id] = visitor_record

                    remember_signature(visitor_record, track_id, frame_num, signature)
                    color = (255, 160, 0)
                    label = f"Visitor #{visitor_record.person_id}"
                else:
                    color = (120, 120, 120)
                    label = f"Ignored ID:{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        present_visitor_ids = {
            record.person_id
            for record in visitor_records
            if frame_num - record.last_seen_frame <= PRESENT_GRACE_FRAMES
        }
        draw_status(
            frame,
            present_count=len(present_visitor_ids),
            total_unique=len(visitor_records),
            staff_count=len(staff_records),
        )

        if not args.no_display:
            cv2.imshow("Visitor Presence Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames > 0 and frame_num >= args.max_frames:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("=" * 60)
    print(f"Frames processed: {frame_num}")
    print(f"Unique visitors seen outside reception area: {len(visitor_records)}")
    print(f"Confirmed staff/receptionists excluded: {len(staff_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
