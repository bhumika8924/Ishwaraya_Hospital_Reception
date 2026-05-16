import cv2
import json
import numpy as np
import os
import time
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────
model = YOLO("yolov8n.pt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.getenv(
    "RECEPTIONIST_VIDEO_PATH",
    os.path.join(BASE_DIR, "D:/main rec/vdo2.mp4")
)
reference_path = "receptionist_uniform_ref.png"
zone_path      = "reception_zone.json"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video: {video_path}")
    raise SystemExit(1)

# Reception desk zone — LEFT side (drag corners with mouse to adjust)
zone_points = np.array([
    [500,  110],
    [1600, 100],
    [1850, 1080],
    [520,  1080],
], dtype=np.int32)

# ─────────────────────────────────────────────
#  TUNABLE SETTINGS
# ─────────────────────────────────────────────
MIN_BOX_HEIGHT  = 80      # ignore tiny detections
MIN_BOX_WIDTH   = 40
MATCH_THRESHOLD = 0.55    # uniform color similarity
MIN_RED_PIXELS  = 20      # red lanyard sensitivity
MIN_ZONE_OVERLAP = 0.30   # count person in zone if 30% of box is inside ROI
CONFIRM_FRAMES_NEEDED = 3 # must match as receptionist this many times before locking in

# Tracking Variables
receptionist_track_ids = set()
receptionist_confirm_count = {}   # {track_id: count_of_receptionist_detections}


# ═════════════════════════════════════════════
#  FUNCTION 1 — Color fingerprint of an image
# ═════════════════════════════════════════════
def make_hs_hist(image):
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# ═════════════════════════════════════════════
#  FUNCTION 2 — Load reference uniform image
# ═════════════════════════════════════════════
def load_reference_hist(path):
    ref_img = cv2.imread(path)
    if ref_img is None:
        raise FileNotFoundError(
            f"Could not read '{path}'. "
            "Save a cropped photo of the receptionist uniform as this file."
        )
    return make_hs_hist(ref_img)


# ═════════════════════════════════════════════
#  FUNCTION 3 — Load reception ROI
# ═════════════════════════════════════════════
def load_zone(path, default_zone):
    try:
        with open(path, "r", encoding="utf-8") as f:
            points = json.load(f)
        zone = np.array(points, dtype=np.int32)
        if zone.shape != (4, 2):
            raise ValueError("Zone must contain exactly four [x, y] points.")
        print(f"Loaded saved reception zone from '{path}': {zone.tolist()}")
        return zone
    except FileNotFoundError:
        print(f"No saved reception zone found. Using default zone.")
        return default_zone
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Could not load '{path}' ({e}). Using default zone.")
        return default_zone


# ═════════════════════════════════════════════
#  FUNCTION 4 — Uniform color match score
#  Looks at the TORSO region (25%–75% of box)
#  Returns score: 1.0 = perfect match, 0 = no match
# ═════════════════════════════════════════════
def outfit_match_score(frame, box, reference_hist):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0

    # Middle band = torso / uniform area
    top    = int(crop.shape[0] * 0.25)
    bottom = int(crop.shape[0] * 0.75)
    torso  = crop[top:bottom]
    if torso.size == 0:
        return -1.0

    person_hist = make_hs_hist(torso)
    return cv2.compareHist(reference_hist, person_hist, cv2.HISTCMP_CORREL)


# ═════════════════════════════════════════════
#  FUNCTION 5 — ID card / lanyard detection
#  Looks at NECK + UPPER CHEST (18%–50% of box)
#  Detects red-pink lanyard color
#  Returns (found: bool, red_pixel_count: int)
# ═════════════════════════════════════════════
def has_id_card_lanyard(frame, box, min_red_pixels=MIN_RED_PIXELS):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0

    # Focus on NECK + UPPER CHEST — where lanyard always hangs
    neck_top    = int(crop.shape[0] * 0.18)
    neck_bottom = int(crop.shape[0] * 0.50)
    neck_region = crop[neck_top:neck_bottom]
    if neck_region.size == 0:
        return False, 0

    hsv = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)

    # Range 1: pure red (lower end of HSV wheel)
    mask1 = cv2.inRange(hsv, np.array([0,   80,  80]),
                             np.array([10,  255, 255]))

    # Range 2: red (upper end of HSV wheel — red wraps around)
    mask2 = cv2.inRange(hsv, np.array([165,  80,  80]),
                             np.array([180, 255, 255]))

    # Range 3: pink-red (matches the lanyard in your CCTV image)
    mask3 = cv2.inRange(hsv, np.array([140,  50, 100]),
                             np.array([165, 255, 255]))

    red_mask        = cv2.bitwise_or(mask1, mask2)
    red_mask        = cv2.bitwise_or(red_mask, mask3)
    red_pixel_count = cv2.countNonZero(red_mask)

    return red_pixel_count >= min_red_pixels, red_pixel_count


# ═════════════════════════════════════════════
#  FUNCTION 6 — Is person inside the zone?
#  Uses how much of the full person box overlaps the ROI
# ═════════════════════════════════════════════
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


# ─────────────────────────────────────────────
#  LOAD REFERENCE + OPEN WINDOW
# ─────────────────────────────────────────────
zone_points = load_zone(zone_path, zone_points)
reference_hist = load_reference_hist(reference_path)

cv2.namedWindow("Receptionist Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Receptionist Monitor", 1280, 720)

print("=" * 55)
print("  Receptionist Monitor — Started")
print("=" * 55)
print("  Q         → Quit")
print("=" * 55)


# ═════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    original_frame = frame.copy()

    # ── Run YOLO person tracking ─────────────
    results = model.track(original_frame, persist=True, verbose=False, conf=0.5, iou=0.4)

    # ── Draw reception zone (yellow) ─────────
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_points], (0, 255, 255))
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 255, 255), thickness=2)

    # ── Counter ─────────────────────────────
    receptionists_in_zone = 0

    # ── Process each detected person ─────────
    boxes = results[0].boxes
    if boxes.id is not None:
        for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
            if int(cls) != 0:          # class 0 = person in YOLO
                continue

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            # Skip boxes that are too small (distant / partial detections)
            if (y2 - y1) < MIN_BOX_HEIGHT or (x2 - x1) < MIN_BOX_WIDTH:
                continue

            # ── Check if person in the zone ──
            in_zone = is_in_zone((x1, y1, x2, y2), zone_points, original_frame.shape)

            # ── Identify Receptionist ──
            is_receptionist = track_id in receptionist_track_ids

            if not is_receptionist:
                # ── Check uniform color match ─────
                score         = outfit_match_score(original_frame, (x1, y1, x2, y2), reference_hist)
                uniform_match = score >= MATCH_THRESHOLD

                # ── Check red ID card lanyard ─────
                id_found, red_count = has_id_card_lanyard(
                    original_frame, (x1, y1, x2, y2), MIN_RED_PIXELS
                )

                # Must be in reception zone AND match uniform/lanyard to be receptionist
                if in_zone and (uniform_match or id_found):
                    # Require multiple confirmations before locking as receptionist
                    receptionist_confirm_count[track_id] = receptionist_confirm_count.get(track_id, 0) + 1

                    if receptionist_confirm_count[track_id] >= CONFIRM_FRAMES_NEEDED:
                        is_receptionist = True
                        receptionist_track_ids.add(track_id)
                        print(f"  [RECEPTIONIST] ID {track_id} confirmed (score={score:.2f}, red={red_count})")

            # ── Count receptionists in zone ──
            if is_receptionist and in_zone:
                receptionists_in_zone += 1

                # Draw bounding box for receptionist
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Recep ID:{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Status banner ─────────────
    if receptionists_in_zone > 0:
        banner_color = (0, 180, 0)
    else:
        banner_color = (0, 0, 210)

    cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (400, 60), banner_color, 3)

    cv2.putText(frame, f"Receptionist at Desk: {receptionists_in_zone}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, banner_color, 3)

    cv2.imshow("Receptionist Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()