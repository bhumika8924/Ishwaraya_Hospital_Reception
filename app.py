"""
Hospital reception monitor.

Detects people with YOLO tracking, confirms receptionists by reception-zone presence
plus uniform-color or lanyard evidence, and counts visitor entry/exit using two
configured crossing lines.
"""
from dataclasses import dataclass, field
import json
import os

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="Hospital Reception Monitor", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
REFERENCE_PATH = os.path.join(BASE_DIR, "receptionist_uniform_ref.png")
ZONE_PATH = os.path.join(BASE_DIR, "reception_zone.json")
TWO_LINE_PATH = os.path.join(BASE_DIR, "two_line_counter_lines.json")

# Defaults match a 1920x1080 layout and are scaled to the selected video.
DEFAULT_RECEPTION_REF = np.array(
    [[500, 110], [1600, 100], [1850, 1080], [520, 1080]], dtype=np.float64
)

MIN_ZONE_OVERLAP = 0.30
CONFIRM_FRAMES_NEEDED = 3
LINE_HIT_DISTANCE = 28
LINE_COOLDOWN_FRAMES = 12
LINE_SEQUENCE_WINDOW_FRAMES = 260
EVENT_MATCH_THRESHOLD = 0.45


@dataclass
class VisitorTrackState:
    last_hit_line: int | None = None
    last_hit_frame: int = -10_000
    crossed_lines: list = field(default_factory=list)
    previous_center: tuple[int, int] | None = None


@dataclass
class VisitorLineEvent:
    event_id: int
    line: int
    track_id: int
    frame_num: int
    signature: np.ndarray | None
    used: bool = False


def make_hs_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def load_reference_hist(path):
    ref_img = cv2.imread(path)
    if ref_img is None:
        raise FileNotFoundError(f"Could not read '{path}'")
    return make_hs_hist(ref_img)


def _as_point_list(data):
    """Accept JSON list of four [x, y] points or a dict with 'points'/'reception'."""
    if isinstance(data, dict):
        pts = data.get("points") or data.get("reception")
        if pts is None:
            raise ValueError("JSON dict must contain 'points' or 'reception'")
        return np.array(pts, dtype=np.float64)
    return np.array(data, dtype=np.float64)


def load_polygon_json(path, default_pts):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        zone = _as_point_list(raw)
        if zone.shape != (4, 2):
            raise ValueError("Zone must have exactly four [x, y] points")
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
    out = []
    for x, y in np.asarray(pts, dtype=np.float64).reshape(-1, 2):
        out.append([int(round(x * sx)), int(round(y * sy))])
    return np.array(out, dtype=np.int32)


def outfit_match_score(frame, box, reference_hist):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0

    top = int(crop.shape[0] * 0.25)
    bottom = int(crop.shape[0] * 0.75)
    torso = crop[top:bottom]
    if torso.size == 0:
        return -1.0

    person_hist = make_hs_hist(torso)
    return cv2.compareHist(reference_hist, person_hist, cv2.HISTCMP_CORREL)


def has_id_card_lanyard(frame, box, min_red_pixels):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

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
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.bitwise_or(red_mask, mask3)
    red_pixel_count = cv2.countNonZero(red_mask)
    return red_pixel_count >= min_red_pixels, red_pixel_count


def is_in_zone(box, poly, frame_shape, min_overlap=MIN_ZONE_OVERLAP):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return False

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [poly], 255)
    person_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)
    overlap_mask = cv2.bitwise_and(roi_mask, person_mask)
    overlap_area = cv2.countNonZero(overlap_mask)
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

    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


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
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def nearest_hit_line(point, lines, previous_point=None):
    crossed = []
    if previous_point is not None and point_distance(previous_point, point) >= 2:
        for idx, line in enumerate(lines):
            if segments_intersect(previous_point, point, line[0], line[1]):
                crossed.append(idx + 1)

    if crossed:
        return crossed[0]

    distances = [point_to_line_distance(point, line) for line in lines]
    best_index = int(np.argmin(distances))
    if distances[best_index] <= LINE_HIT_DISTANCE:
        return best_index + 1
    return None


def person_signature(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
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

        score = compare_signatures(signature, event.signature)
        if score < 0:
            score = 0.0

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


def draw_reception_badge(frame, receptionists_in_zone):
    banner_color = (0, 180, 0) if receptionists_in_zone > 0 else (0, 0, 210)
    cv2.rectangle(frame, (0, 0), (460, 58), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (460, 58), banner_color, 2)
    cv2.putText(
        frame,
        f"Receptionists at desk: {receptionists_in_zone}",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        banner_color,
        2,
    )


def draw_visitor_badge(frame, entry_count, exit_count, entry_order, exit_order):
    x0 = max(frame.shape[1] - 500, 0)
    cv2.rectangle(frame, (x0, 0), (frame.shape[1], 84), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, 0), (frame.shape[1], 84), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"Entries L{entry_order[0]}->L{entry_order[1]}: {entry_count}",
        (x0 + 12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Exits L{exit_order[0]}->L{exit_order[1]}: {exit_count}",
        (x0 + 12, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (220, 220, 220),
        2,
    )


def draw_visitor_lines(frame, lines):
    cv2.line(frame, lines[0][0], lines[0][1], (0, 255, 255), 3)
    cv2.putText(
        frame,
        "Line 1",
        (lines[0][0][0] + 8, max(lines[0][0][1] - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.line(frame, lines[1][0], lines[1][1], (255, 0, 255), 3)
    cv2.putText(
        frame,
        "Line 2",
        (lines[1][0][0] + 8, max(lines[1][0][1] - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2,
    )


def resize_for_preview(frame, max_width):
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    preview_h = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_width, preview_h), interpolation=cv2.INTER_AREA)


@st.cache_resource
def load_model(path):
    return YOLO(path)


st.title("Hospital reception monitor")
st.caption(
    "Counts confirmed receptionists at the desk and visitor entry/exit crossings "
    "from the saved two-line setup."
)

if not os.path.isfile(MODEL_PATH):
    st.error(f"Place YOLO weights at `{MODEL_PATH}`.")
    st.stop()

if not os.path.isfile(REFERENCE_PATH):
    st.error(f"Uniform reference image missing: `{REFERENCE_PATH}`")
    st.stop()

try:
    reference_hist = load_reference_hist(REFERENCE_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

model = load_model(MODEL_PATH)
reception_ref = load_polygon_json(ZONE_PATH, DEFAULT_RECEPTION_REF)

with st.sidebar:
    st.header("Video and layout")
    default_video = os.path.join(BASE_DIR, "five_min_vdo.mp4")
    video_path = st.text_input(
        "Video file path",
        value=default_video,
        help="Any file from this camera. Use forward slashes or escaped backslashes on Windows.",
    )
    video_path = os.path.normpath(os.path.expanduser(video_path.strip()))
    ref_w = st.number_input(
        "Reference width (px zone was drawn for)",
        min_value=320,
        max_value=7680,
        value=1920,
        step=1,
    )
    ref_h = st.number_input(
        "Reference height",
        min_value=240,
        max_value=4320,
        value=1080,
        step=1,
    )

    st.markdown("**Detection tuning**")
    yolo_conf = st.slider("YOLO confidence", 0.15, 0.95, 0.50, 0.05)
    yolo_iou = st.slider("YOLO IoU", 0.20, 0.80, 0.40, 0.05)
    match_threshold = st.slider("Uniform match threshold", 0.20, 0.90, 0.55, 0.01)
    min_red = st.slider("Min red lanyard pixels", 5, 120, 20, 1)
    min_overlap = st.slider("Min zone overlap", 0.10, 0.90, MIN_ZONE_OVERLAP, 0.05)
    st.markdown("**Visitor two-line counter**")
    two_line_path = st.text_input(
        "Two-line JSON path",
        value=TWO_LINE_PATH,
        help="Created by two_line_visitor_counter.py.",
    )
    two_line_path = os.path.normpath(os.path.expanduser(two_line_path.strip()))
    entry_order_option = st.selectbox(
        "Entry direction",
        ["1-2", "2-1"],
        index=0,
        help="Use 1-2 when crossing Line 1 then Line 2 means entry. Use 2-1 for the opposite.",
    )
    st.markdown("**Speed**")
    ui_update_every = st.slider("Preview update every N frames", 5, 120, 5, 5)
    preview_width = st.slider("Preview width", 480, 1280, 900, 20)
    analyze_every = st.slider(
        "Analyze every Nth frame",
        1,
        5,
        1,
        help="Keep this at 1 for best accuracy. Increase only when you need faster processing.",
    )

st.markdown("---")

if st.button("Run reception monitor", type="primary"):
    if not os.path.isfile(video_path):
        st.error(f"Video not found: `{video_path}`")
        st.stop()
    if not os.path.isfile(two_line_path):
        st.error(
            f"Two-line file not found: `{two_line_path}`. "
            "Run `python two_line_visitor_counter.py --redraw` first."
        )
        st.stop()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: `{video_path}`")
        st.stop()

    ret0, frame0 = cap.read()
    if not ret0 or frame0 is None:
        cap.release()
        st.error("Could not read the first frame.")
        st.stop()

    fh, fw = frame0.shape[:2]
    zone_points = scale_points(reception_ref, ref_w, ref_h, fw, fh)
    try:
        visitor_lines = load_two_line_json(two_line_path, fw, fh)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        cap.release()
        st.error(f"Could not load two-line file: {exc}")
        st.stop()

    min_box_h = max(36, int(round(80 * fh / float(ref_h))))
    min_box_w = max(20, int(round(40 * fw / float(ref_w))))
    visitor_min_box_h = max(36, int(round(70 * fh / 720.0)))
    visitor_min_box_w = max(20, int(round(32 * fw / 1280.0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    progress_bar = st.progress(0, text="Starting...")
    status = st.empty()
    frame_preview = st.empty()
    metrics = st.empty()

    receptionist_track_ids = set()
    receptionist_confirm_count = {}
    max_receptionists = 0
    visitor_track_states = {}
    visitor_line_events = []
    next_visitor_event_id = 1
    counted_entry_ids = set()
    counted_exit_ids = set()
    visitor_entry_count = 0
    visitor_exit_count = 0
    visitor_entry_order = entry_sequence(entry_order_option)
    visitor_exit_order = exit_sequence(entry_order_option)
    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if analyze_every > 1 and (frame_num - 1) % analyze_every != 0:
            if total_frames > 0 and frame_num % ui_update_every == 0:
                progress_bar.progress(
                    min(frame_num / total_frames, 1.0),
                    text=f"Frame {frame_num} / {total_frames}",
                )
            continue

        original_frame = frame.copy()

        results = model.track(
            original_frame,
            persist=True,
            verbose=False,
            conf=yolo_conf,
            iou=yolo_iou,
        )

        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_points], (0, 255, 255))
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 255, 255), thickness=2)
        draw_visitor_lines(frame, visitor_lines)

        receptionists_in_zone = 0
        total_people_in_frame = 0

        boxes = results[0].boxes
        if boxes.id is not None:
            for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
                if int(cls) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                if (y2 - y1) < min_box_h or (x2 - x1) < min_box_w:
                    continue

                total_people_in_frame += 1
                in_zone = is_in_zone(
                    (x1, y1, x2, y2),
                    zone_points,
                    original_frame.shape,
                    min_overlap=min_overlap,
                )

                if track_id not in receptionist_track_ids:
                    score = outfit_match_score(
                        original_frame, (x1, y1, x2, y2), reference_hist
                    )
                    uniform_match = score >= match_threshold
                    id_found, _ = has_id_card_lanyard(
                        original_frame, (x1, y1, x2, y2), min_red
                    )

                    if in_zone and (uniform_match or id_found):
                        receptionist_confirm_count[track_id] = (
                            receptionist_confirm_count.get(track_id, 0) + 1
                        )
                        if receptionist_confirm_count[track_id] >= CONFIRM_FRAMES_NEEDED:
                            receptionist_track_ids.add(track_id)
                    else:
                        receptionist_confirm_count.pop(track_id, None)

                if track_id in receptionist_track_ids:
                    if in_zone:
                        receptionists_in_zone += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(
                        frame,
                        f"Recep ID:{track_id}",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                elif in_zone:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                    cv2.putText(
                        frame,
                        f"Checking ID:{track_id}",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 255),
                        2,
                    )

                if (
                    track_id not in receptionist_track_ids
                    and (y2 - y1) >= visitor_min_box_h
                    and (x2 - x1) >= visitor_min_box_w
                ):
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    signature = person_signature(original_frame, (x1, y1, x2, y2))
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
                        and frame_num - visitor_state.last_hit_frame >= LINE_COOLDOWN_FRAMES
                    ):
                        visitor_state.crossed_lines.append(hit_line)
                        visitor_state.crossed_lines = visitor_state.crossed_lines[-2:]
                        visitor_state.last_hit_line = hit_line
                        visitor_state.last_hit_frame = frame_num

                        visitor_line_events.append(
                            VisitorLineEvent(
                                event_id=next_visitor_event_id,
                                line=hit_line,
                                track_id=track_id,
                                frame_num=frame_num,
                                signature=signature,
                            )
                        )
                        next_visitor_event_id += 1

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
                    cv2.circle(frame, center_point, 4, (0, 0, 255), -1)

        visitor_line_events = [
            event
            for event in visitor_line_events
            if frame_num - event.frame_num <= LINE_SEQUENCE_WINDOW_FRAMES
        ]

        max_receptionists = max(max_receptionists, receptionists_in_zone)

        draw_reception_badge(frame, receptionists_in_zone)
        draw_visitor_badge(
            frame,
            visitor_entry_count,
            visitor_exit_count,
            visitor_entry_order,
            visitor_exit_order,
        )
        cv2.putText(
            frame,
            f"In frame: {total_people_in_frame} | Confirmed receptionist IDs: {len(receptionist_track_ids)}",
            (10, fh - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        last_frame = frame.copy()

        if frame_num % ui_update_every == 0 or frame_num == 1:
            if total_frames > 0:
                progress_bar.progress(
                    min(frame_num / total_frames, 1.0),
                    text=f"Frame {frame_num} / {total_frames}",
                )
            else:
                progress_bar.progress(0, text=f"Frame {frame_num}")
            status.write(
                f"Frame **{frame_num}** - receptionists: **{receptionists_in_zone}**, "
                f"visitor entries: **{visitor_entry_count}**, exits: **{visitor_exit_count}**"
            )
            left_metric, right_metric = metrics.columns(2)
            left_metric.metric("Receptionists at desk", receptionists_in_zone)
            right_metric.metric(
                "Visitor entry / exit",
                f"{visitor_entry_count} / {visitor_exit_count}",
            )
            preview = resize_for_preview(frame, preview_width)
            frame_preview.image(preview, channels="BGR", width="stretch")

    cap.release()
    progress_bar.progress(1.0, text="Done")

    if last_frame is not None:
        preview = resize_for_preview(last_frame, preview_width)
        frame_preview.image(preview, channels="BGR", width="stretch")

    status.write("**Analysis complete.**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Frames processed", frame_num)
    with c2:
        st.metric("Peak receptionists at desk", max_receptionists)
    with c3:
        st.metric("Confirmed receptionist IDs", len(receptionist_track_ids))
    with c4:
        st.metric(
            f"Visitor entries L{visitor_entry_order[0]}->L{visitor_entry_order[1]}",
            visitor_entry_count,
        )
    with c5:
        st.metric(
            f"Visitor exits L{visitor_exit_order[0]}->L{visitor_exit_order[1]}",
            visitor_exit_count,
        )

    st.success(
        f"Finished. Peak receptionists at desk: {max_receptionists}. "
        f"Visitor entries: {visitor_entry_count}, exits: {visitor_exit_count}."
    )

st.markdown("---")
st.caption(
    "`reception_zone.json` controls the four-point reception desk zone. "
    "`receptionist_uniform_ref.png` controls the uniform color match. "
    "`two_line_counter_lines.json` controls visitor entry/exit lines."
)
