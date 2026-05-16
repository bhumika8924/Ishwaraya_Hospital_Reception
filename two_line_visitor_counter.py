"""
Two-line visitor entry counter.

Workflow:
1. Draw Line 1, then Line 2 on the first video frame.
2. A person crossing Line 1 -> Line 2 is counted as an entry.
3. A person crossing Line 2 -> Line 1 is treated as exit and is not counted.
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "vdo2.mp4")
LINES_PATH = os.path.join(BASE_DIR, "two_line_counter_lines.json")

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

YOLO_CONF = 0.35
YOLO_IOU = 0.45
MIN_BOX_HEIGHT = 70
MIN_BOX_WIDTH = 32
LINE_HIT_DISTANCE = 28
LINE_COOLDOWN_FRAMES = 12
LINE_SEQUENCE_WINDOW_FRAMES = 260
EVENT_MATCH_THRESHOLD = 0.45


@dataclass
class TrackState:
    last_hit_line: int | None = None
    last_hit_frame: int = -10_000
    crossed_lines: list = field(default_factory=list)
    previous_center: tuple[int, int] | None = None


@dataclass
class LineEvent:
    event_id: int
    line: int
    track_id: int
    frame_num: int
    signature: np.ndarray | None
    used: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Two-line visitor entry counter.")
    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Video path. Default is five_min_vdo.mp4 in this folder.",
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="Ignore saved lines and draw Line 1/Line 2 again.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening the video preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for testing. 0 means full video.",
    )
    parser.add_argument(
        "--entry-order",
        choices=("1-2", "2-1"),
        default="1-2",
        help="Which crossing order means entry. Default: 1-2.",
    )
    return parser.parse_args()


class LineDrawer:
    def __init__(self, frame):
        self.frame = frame
        self.lines = []
        self.current_start = None
        self.preview_point = None
        self.window_name = "Draw Line 1 and Line 2"

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.lines) < 2:
            self.current_start = (x, y)
            self.preview_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.current_start is not None:
            self.preview_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.current_start is not None:
            end = (x, y)
            if distance(self.current_start, end) >= 20:
                self.lines.append([self.current_start, end])
            self.current_start = None
            self.preview_point = None

    def draw(self):
        canvas = self.frame.copy()
        colors = [(0, 255, 255), (255, 0, 255)]

        for idx, line in enumerate(self.lines):
            p1, p2 = line
            cv2.line(canvas, p1, p2, colors[idx], 3)
            put_label(canvas, f"Line {idx + 1}", p1, colors[idx])

        if self.current_start is not None and self.preview_point is not None:
            color = colors[len(self.lines)]
            cv2.line(canvas, self.current_start, self.preview_point, color, 2)
            put_label(canvas, f"Line {len(self.lines) + 1}", self.current_start, color)

        cv2.rectangle(canvas, (0, 0), (760, 96), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            "Draw Line 1 then Line 2: drag mouse left button",
            (14, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "ENTER=start  R=reset  S=save  Q=quit",
            (14, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (220, 220, 220),
            2,
        )
        return canvas

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            cv2.imshow(self.window_name, self.draw())
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyWindow(self.window_name)
                raise SystemExit(0)
            if key == ord("r"):
                self.lines.clear()
                self.current_start = None
                self.preview_point = None
            if key == ord("s") and len(self.lines) == 2:
                save_lines(self.lines)
            if key in (13, 10) and len(self.lines) == 2:
                cv2.destroyWindow(self.window_name)
                return self.lines


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def put_label(frame, text, point, color):
    x, y = point
    cv2.putText(
        frame,
        text,
        (x + 8, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def save_lines(lines):
    data = {
        "reference_width": FRAME_WIDTH,
        "reference_height": FRAME_HEIGHT,
        "lines": lines,
    }
    with open(LINES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved lines to {LINES_PATH}")


def load_lines():
    if not os.path.isfile(LINES_PATH):
        return None

    try:
        with open(LINES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = data["lines"]
        if len(lines) != 2:
            raise ValueError("Need exactly two lines")

        ref_w = float(data.get("reference_width", FRAME_WIDTH))
        ref_h = float(data.get("reference_height", FRAME_HEIGHT))
        sx = FRAME_WIDTH / ref_w
        sy = FRAME_HEIGHT / ref_h

        scaled = []
        for line in lines:
            p1, p2 = line
            scaled.append([
                (int(round(p1[0] * sx)), int(round(p1[1] * sy))),
                (int(round(p2[0] * sx)), int(round(p2[1] * sy))),
            ])
        return scaled
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not load saved lines: {exc}")
        return None


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
    if previous_point is not None and distance(previous_point, point) >= 2:
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


def find_recent_event(events, target_line, frame_num, signature):
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


def draw_count_overlay(frame, entry_count, exit_count, entry_order, exit_order):
    cv2.rectangle(frame, (0, 0), (500, 96), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Entry Count: {entry_count}",
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Exit Count: {exit_count}",
        (14, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")

    first_frame = cv2.resize(first_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    lines = None if args.redraw else load_lines()
    if lines is None:
        lines = LineDrawer(first_frame).run()
        save_lines(lines)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    model = YOLO(MODEL_PATH)

    if not args.no_display:
        cv2.namedWindow("Two Line Visitor Counter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Two Line Visitor Counter", FRAME_WIDTH, FRAME_HEIGHT)

    track_states = {}
    counted_entry_ids = set()
    counted_exit_ids = set()
    line_events = []
    next_event_id = 1
    entry_count = 0
    exit_count = 0
    frame_num = 0
    entry_order = entry_sequence(args.entry_order)
    exit_order = exit_sequence(args.entry_order)

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

        cv2.line(frame, lines[0][0], lines[0][1], (0, 255, 255), 3)
        put_label(frame, "Line 1", lines[0][0], (0, 255, 255))
        cv2.line(frame, lines[1][0], lines[1][1], (255, 0, 255), 3)
        put_label(frame, "Line 2", lines[1][0], (255, 0, 255))

        boxes = results[0].boxes
        if boxes.id is not None:
            for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
                if int(cls) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                if (y2 - y1) < MIN_BOX_HEIGHT or (x2 - x1) < MIN_BOX_WIDTH:
                    continue

                signature = person_signature(frame, (x1, y1, x2, y2))
                center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                state = track_states.setdefault(track_id, TrackState())
                hit_line = nearest_hit_line(
                    center_point,
                    lines,
                    previous_point=state.previous_center,
                )

                if (
                    hit_line is not None
                    and hit_line != state.last_hit_line
                    and frame_num - state.last_hit_frame >= LINE_COOLDOWN_FRAMES
                ):
                    state.crossed_lines.append(hit_line)
                    state.crossed_lines = state.crossed_lines[-2:]
                    state.last_hit_line = hit_line
                    state.last_hit_frame = frame_num
                    line_events.append(
                        LineEvent(
                            event_id=next_event_id,
                            line=hit_line,
                            track_id=track_id,
                            frame_num=frame_num,
                            signature=signature,
                        )
                    )
                    next_event_id += 1

                    if state.crossed_lines == entry_order and track_id not in counted_entry_ids:
                        entry_count += 1
                        counted_entry_ids.add(track_id)
                        print(
                            f"[ENTRY] ID {track_id}: Line {entry_order[0]} -> "
                            f"Line {entry_order[1]}. Total={entry_count}"
                        )
                    elif hit_line == entry_order[1] and track_id not in counted_entry_ids:
                        previous_line1 = find_recent_event(
                            line_events[:-1],
                            target_line=entry_order[0],
                            frame_num=frame_num,
                            signature=signature,
                        )
                        if previous_line1 is not None:
                            entry_count += 1
                            counted_entry_ids.add(track_id)
                            previous_line1.used = True
                            line_events[-1].used = True
                            print(
                                f"[ENTRY] ID {previous_line1.track_id}->{track_id}: "
                                f"Line {entry_order[0]} -> Line {entry_order[1]}. "
                                f"Total={entry_count}"
                            )

                    if state.crossed_lines == exit_order and track_id not in counted_exit_ids:
                        exit_count += 1
                        counted_exit_ids.add(track_id)
                        print(
                            f"[EXIT] ID {track_id}: Line {exit_order[0]} -> "
                            f"Line {exit_order[1]}. Entries stay={entry_count}"
                        )
                    elif hit_line == exit_order[1] and track_id not in counted_exit_ids:
                        previous_line2 = find_recent_event(
                            line_events[:-1],
                            target_line=exit_order[0],
                            frame_num=frame_num,
                            signature=signature,
                        )
                        if previous_line2 is not None:
                            exit_count += 1
                            counted_exit_ids.add(track_id)
                            previous_line2.used = True
                            line_events[-1].used = True
                            print(
                                f"[EXIT] ID {previous_line2.track_id}->{track_id}: "
                                f"Line {exit_order[0]} -> Line {exit_order[1]}. "
                                f"Entries stay={entry_count}"
                            )

                line_events = [
                    event
                    for event in line_events
                    if frame_num - event.frame_num <= LINE_SEQUENCE_WINDOW_FRAMES
                ]

                color = (255, 160, 0) if track_id in counted_entry_ids else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                state.previous_center = center_point

        draw_count_overlay(frame, entry_count, exit_count, entry_order, exit_order)
        if not args.no_display:
            cv2.imshow("Two Line Visitor Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames > 0 and frame_num >= args.max_frames:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("=" * 60)
    print(f"Final entry count Line {entry_order[0]} -> Line {entry_order[1]}: {entry_count}")
    print(f"Exit crossings Line {exit_order[0]} -> Line {exit_order[1]}: {exit_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
