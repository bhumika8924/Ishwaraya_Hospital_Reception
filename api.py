"""
FastAPI service for reception and visitor counts.

Run:
    uvicorn api:app --reload

Open:
    http://localhost:8000/docs
"""
from dataclasses import dataclass, field
import json
import os
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
REFERENCE_PATH = os.path.join(BASE_DIR, "receptionist_uniform_ref.png")
ZONE_PATH = os.path.join(BASE_DIR, "reception_zone.json")
TWO_LINE_PATH = os.path.join(BASE_DIR, "two_line_counter_lines.json")

DEFAULT_RECEPTION_REF = np.array(
    [[500, 110], [1600, 100], [1850, 1080], [520, 1080]], dtype=np.float64
)

MIN_ZONE_OVERLAP = 0.30
CONFIRM_FRAMES_NEEDED = 3
LINE_HIT_DISTANCE = 28
LINE_COOLDOWN_FRAMES = 12
LINE_SEQUENCE_WINDOW_FRAMES = 260
EVENT_MATCH_THRESHOLD = 0.45

app = FastAPI(title="Hospital Reception Counter API")
_model = None
_reference_hist = None


class AnalyzeRequest(BaseModel):
    video_path: str = Field(default=os.path.join(BASE_DIR, "five_min_vdo.mp4"))
    two_line_path: str = Field(default=TWO_LINE_PATH)
    zone_path: str = Field(default=ZONE_PATH)
    ref_w: int = Field(default=1920, ge=320)
    ref_h: int = Field(default=1080, ge=240)
    entry_order: Literal["1-2", "2-1"] = "1-2"
    yolo_conf: float = Field(default=0.50, ge=0.01, le=1.0)
    yolo_iou: float = Field(default=0.40, ge=0.01, le=1.0)
    match_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    min_red: int = Field(default=20, ge=0)
    min_overlap: float = Field(default=MIN_ZONE_OVERLAP, ge=0.0, le=1.0)
    analyze_every: int = Field(default=1, ge=1, le=10)


@dataclass
class VisitorTrackState:
    last_hit_line: int | None = None
    last_hit_frame: int = -10_000
    crossed_lines: list = field(default_factory=list)
    previous_center: tuple[int, int] | None = None


@dataclass
class VisitorLineEvent:
    line: int
    frame_num: int
    signature: np.ndarray | None
    used: bool = False


def get_model():
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"YOLO weights missing: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
    return _model


def make_hs_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def get_reference_hist():
    global _reference_hist
    if _reference_hist is None:
        ref_img = cv2.imread(REFERENCE_PATH)
        if ref_img is None:
            raise FileNotFoundError(f"Reference image missing: {REFERENCE_PATH}")
        _reference_hist = make_hs_hist(ref_img)
    return _reference_hist


def load_polygon_json(path, default_pts):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        pts = raw.get("points") or raw.get("reception") if isinstance(raw, dict) else raw
        zone = np.array(pts, dtype=np.float64)
        if zone.shape != (4, 2):
            raise ValueError
        return zone
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return np.array(default_pts, copy=True)


def load_two_line_json(path, frame_w, frame_h):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = data["lines"]
    if len(lines) != 2:
        raise ValueError("Line JSON must contain exactly two lines")

    ref_w = float(data.get("reference_width", frame_w))
    ref_h = float(data.get("reference_height", frame_h))
    sx = frame_w / ref_w
    sy = frame_h / ref_h

    scaled = []
    for line in lines:
        p1, p2 = line
        scaled.append([
            (int(round(p1[0] * sx)), int(round(p1[1] * sy))),
            (int(round(p2[0] * sx)), int(round(p2[1] * sy))),
        ])
    return scaled


def scale_points(pts, ref_w, ref_h, frame_w, frame_h):
    sx = frame_w / float(ref_w)
    sy = frame_h / float(ref_h)
    return np.array(
        [[int(round(x * sx)), int(round(y * sy))] for x, y in pts],
        dtype=np.int32,
    )


def outfit_match_score(frame, box, reference_hist):
    x1, y1, x2, y2 = clip_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0

    top = int(crop.shape[0] * 0.25)
    bottom = int(crop.shape[0] * 0.75)
    torso = crop[top:bottom]
    if torso.size == 0:
        return -1.0
    return cv2.compareHist(reference_hist, make_hs_hist(torso), cv2.HISTCMP_CORREL)


def has_id_card_lanyard(frame, box, min_red_pixels):
    x1, y1, x2, y2 = clip_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    neck = crop[int(crop.shape[0] * 0.18): int(crop.shape[0] * 0.50)]
    if neck.size == 0:
        return False

    hsv = cv2.cvtColor(neck, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([140, 50, 100]), np.array([165, 255, 255]))
    red_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
    return cv2.countNonZero(red_mask) >= min_red_pixels


def clip_box(box, frame_shape):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def is_in_zone(box, poly, frame_shape, min_overlap):
    x1, y1, x2, y2 = clip_box(box, frame_shape)
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return False

    h, w = frame_shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    person_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [poly], 255)
    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)
    overlap_area = cv2.countNonZero(cv2.bitwise_and(roi_mask, person_mask))
    return (overlap_area / box_area) >= min_overlap


def point_distance(p1, p2):
    return float(
        np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32))
    )


def point_to_line_distance(point, line):
    p = np.array(point, dtype=np.float32)
    a = np.array(line[0], dtype=np.float32)
    b = np.array(line[1], dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0:
        return float("inf")
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


def orientation(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (o1 == 0 and on_segment(p1, p2, q1))
        or (o2 == 0 and on_segment(p1, q2, q1))
        or (o3 == 0 and on_segment(p2, p1, q2))
        or (o4 == 0 and on_segment(p2, q1, q2))
    )


def nearest_hit_line(point, lines, previous_point=None):
    if previous_point is not None and point_distance(previous_point, point) >= 2:
        for idx, line in enumerate(lines):
            if segments_intersect(previous_point, point, line[0], line[1]):
                return idx + 1

    distances = [point_to_line_distance(point, line) for line in lines]
    best_index = int(np.argmin(distances))
    if distances[best_index] <= LINE_HIT_DISTANCE:
        return best_index + 1
    return None


def person_signature(frame, box):
    x1, y1, x2, y2 = clip_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    body = crop[int(crop.shape[0] * 0.10): int(crop.shape[0] * 0.90)]
    if body.size == 0:
        return None

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compare_signatures(signature_a, signature_b):
    if signature_a is None or signature_b is None:
        return -1.0
    return cv2.compareHist(signature_a, signature_b, cv2.HISTCMP_CORREL)


def find_recent_line_event(events, target_line, frame_num, signature):
    best_event = None
    best_score = -1.0
    best_age = 10_000

    for event in events:
        age = frame_num - event.frame_num
        if event.used or event.line != target_line:
            continue
        if age < 0 or age > LINE_SEQUENCE_WINDOW_FRAMES:
            continue

        score = max(compare_signatures(signature, event.signature), 0.0)
        if score > best_score or (score == best_score and age < best_age):
            best_event = event
            best_score = score
            best_age = age

    if best_event is None:
        return None
    if best_score >= EVENT_MATCH_THRESHOLD or best_age <= 90:
        return best_event
    return None


def entry_sequence(entry_order):
    return [1, 2] if entry_order == "1-2" else [2, 1]


def exit_sequence(entry_order):
    return [2, 1] if entry_order == "1-2" else [1, 2]


def analyze_counts(settings: AnalyzeRequest):
    video_path = os.path.normpath(os.path.expanduser(settings.video_path))
    two_line_path = os.path.normpath(os.path.expanduser(settings.two_line_path))
    zone_path = os.path.normpath(os.path.expanduser(settings.zone_path))

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(two_line_path):
        raise FileNotFoundError(f"Two-line JSON not found: {two_line_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret0, frame0 = cap.read()
    if not ret0 or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read the first video frame.")

    fh, fw = frame0.shape[:2]
    reception_ref = load_polygon_json(zone_path, DEFAULT_RECEPTION_REF)
    zone_points = scale_points(reception_ref, settings.ref_w, settings.ref_h, fw, fh)
    visitor_lines = load_two_line_json(two_line_path, fw, fh)

    min_box_h = max(36, int(round(80 * fh / float(settings.ref_h))))
    min_box_w = max(20, int(round(40 * fw / float(settings.ref_w))))
    visitor_min_box_h = max(36, int(round(70 * fh / 720.0)))
    visitor_min_box_w = max(20, int(round(32 * fw / 1280.0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    model = get_model()
    model.predictor = None
    reference_hist = get_reference_hist()

    receptionist_track_ids = set()
    receptionist_confirm_count = {}
    max_receptionists = 0
    visitor_track_states = {}
    visitor_line_events = []
    counted_entry_ids = set()
    counted_exit_ids = set()
    visitor_entry_count = 0
    visitor_exit_count = 0
    visitor_entry_order = entry_sequence(settings.entry_order)
    visitor_exit_order = exit_sequence(settings.entry_order)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if settings.analyze_every > 1 and (frame_num - 1) % settings.analyze_every != 0:
            continue

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=settings.yolo_conf,
            iou=settings.yolo_iou,
        )

        receptionists_in_zone = 0
        boxes = results[0].boxes
        if boxes.id is not None:
            for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
                if int(cls) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                if (y2 - y1) < min_box_h or (x2 - x1) < min_box_w:
                    continue

                person_box = (x1, y1, x2, y2)
                in_zone = is_in_zone(
                    person_box,
                    zone_points,
                    frame.shape,
                    min_overlap=settings.min_overlap,
                )

                if track_id not in receptionist_track_ids:
                    uniform_match = (
                        outfit_match_score(frame, person_box, reference_hist)
                        >= settings.match_threshold
                    )
                    id_found = has_id_card_lanyard(frame, person_box, settings.min_red)
                    if in_zone and (uniform_match or id_found):
                        receptionist_confirm_count[track_id] = (
                            receptionist_confirm_count.get(track_id, 0) + 1
                        )
                        if receptionist_confirm_count[track_id] >= CONFIRM_FRAMES_NEEDED:
                            receptionist_track_ids.add(track_id)
                    else:
                        receptionist_confirm_count.pop(track_id, None)

                if track_id in receptionist_track_ids and in_zone:
                    receptionists_in_zone += 1

                if (
                    track_id not in receptionist_track_ids
                    and (y2 - y1) >= visitor_min_box_h
                    and (x2 - x1) >= visitor_min_box_w
                ):
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    signature = person_signature(frame, person_box)
                    visitor_state = visitor_track_states.setdefault(
                        track_id, VisitorTrackState()
                    )
                    hit_line = nearest_hit_line(
                        center_point,
                        visitor_lines,
                        previous_point=visitor_state.previous_center,
                    )

                    if (
                        hit_line is not None
                        and hit_line != visitor_state.last_hit_line
                        and frame_num - visitor_state.last_hit_frame
                        >= LINE_COOLDOWN_FRAMES
                    ):
                        visitor_state.crossed_lines.append(hit_line)
                        visitor_state.crossed_lines = visitor_state.crossed_lines[-2:]
                        visitor_state.last_hit_line = hit_line
                        visitor_state.last_hit_frame = frame_num

                        visitor_line_events.append(
                            VisitorLineEvent(
                                line=hit_line,
                                frame_num=frame_num,
                                signature=signature,
                            )
                        )

                        if (
                            visitor_state.crossed_lines == visitor_entry_order
                            and track_id not in counted_entry_ids
                        ):
                            visitor_entry_count += 1
                            counted_entry_ids.add(track_id)
                            visitor_line_events[-1].used = True
                        elif (
                            hit_line == visitor_entry_order[1]
                            and track_id not in counted_entry_ids
                        ):
                            previous_entry_event = find_recent_line_event(
                                visitor_line_events[:-1],
                                target_line=visitor_entry_order[0],
                                frame_num=frame_num,
                                signature=signature,
                            )
                            if previous_entry_event is not None:
                                visitor_entry_count += 1
                                counted_entry_ids.add(track_id)
                                previous_entry_event.used = True
                                visitor_line_events[-1].used = True

                        if (
                            visitor_state.crossed_lines == visitor_exit_order
                            and track_id not in counted_exit_ids
                        ):
                            visitor_exit_count += 1
                            counted_exit_ids.add(track_id)
                            visitor_line_events[-1].used = True
                        elif (
                            hit_line == visitor_exit_order[1]
                            and track_id not in counted_exit_ids
                        ):
                            previous_exit_event = find_recent_line_event(
                                visitor_line_events[:-1],
                                target_line=visitor_exit_order[0],
                                frame_num=frame_num,
                                signature=signature,
                            )
                            if previous_exit_event is not None:
                                visitor_exit_count += 1
                                counted_exit_ids.add(track_id)
                                previous_exit_event.used = True
                                visitor_line_events[-1].used = True

                    visitor_state.previous_center = center_point

        visitor_line_events = [
            event
            for event in visitor_line_events
            if frame_num - event.frame_num <= LINE_SEQUENCE_WINDOW_FRAMES
        ]
        max_receptionists = max(max_receptionists, receptionists_in_zone)

    cap.release()
    return {
        "receptionist_count": max_receptionists,
        "visitor_entries": visitor_entry_count,
        "visitor_exits": visitor_exit_count,
    }


def run_or_400(request: AnalyzeRequest):
    try:
        return analyze_counts(request)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/receptionist-count")
def receptionist_count(request: AnalyzeRequest):
    result = run_or_400(request)
    return {"receptionist_count": result["receptionist_count"]}


@app.post("/visitor-entry-count")
def visitor_entry_count(request: AnalyzeRequest):
    result = run_or_400(request)
    return {"visitor_entries": result["visitor_entries"]}


@app.post("/visitor-exit-count")
def visitor_exit_count(request: AnalyzeRequest):
    result = run_or_400(request)
    return {"visitor_exits": result["visitor_exits"]}
