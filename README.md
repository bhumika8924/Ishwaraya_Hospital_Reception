# Hospital Reception Monitor

Streamlit and OpenCV tools for monitoring a hospital reception desk.
The project detects people with YOLO, counts confirmed receptionists in the
configured reception zone, and counts unique visitors present outside the
receptionist/reception area while excluding confirmed staff.

## Project Files

```text
app.py                          <- Main Streamlit receptionist counter
footfall_counter.py             <- Local OpenCV unique visitor presence counter
two_line_visitor_counter.py     <- Draw two lines and count Line 1 -> Line 2 entries
receptionist_count.py           <- Local OpenCV receptionist counter
requirements.txt                <- Python dependencies
yolov8n.pt                      <- YOLOv8 model weights
receptionist_uniform_ref.png    <- Reference image for receptionist uniform
reception_zone.json             <- Four-point reception desk zone
visitor_entry_zone.json         <- Four-point visitor entrance zone
```

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at:

```text
http://localhost:8501
```

Run the visitor presence counter:

```bash
python footfall_counter.py
```

For a headless test:

```bash
python footfall_counter.py --no-display --max-frames 2800
```

Run the two-line visitor entry counter:

```bash
python two_line_visitor_counter.py
```

Redraw saved lines:

```bash
python two_line_visitor_counter.py --redraw
```

## How It Works

- YOLO detects people in each video frame.
- Tracking keeps a stable ID for each detected person.
- The reception-zone polygon decides whether a person is at the desk.
- The torso region is compared with `receptionist_uniform_ref.png`.
- The neck/upper-chest region is checked for red or pink lanyard pixels.
- A person must match for multiple frames before being confirmed as receptionist.
- Visitors are counted only when they are outside the reception/receptionist zone.
- People inside `reception_zone.json` are excluded from the visitor count.
- A full-session appearance memory suppresses duplicate visitor counts when the
  tracker changes IDs.

## Main Metrics

- Receptionists currently at desk
- Peak receptionists at desk
- Confirmed receptionist IDs
- Visitors currently present outside reception area
- Unique visitors seen outside reception area
- Frames processed

## Calibration Files

- Edit `reception_zone.json` when the yellow reception zone is not aligned with the desk.
- Replace `receptionist_uniform_ref.png` when the uniform color/reference changes.
