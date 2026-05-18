# Hospital Reception Monitor

Streamlit and OpenCV tools for monitoring a hospital reception desk.
The project detects people with YOLO, counts confirmed receptionists in the
configured reception zone, and counts visitor entries/exits using two saved
crossing lines while excluding confirmed staff from visitor counts.

## Project Files

```text
dual_counter_viewer.py          <- Main Streamlit app for receptionist + entry/exit counts
app.py                          <- Streamlit reference app for receptionist count workflow
footfall_counter.py             <- Local OpenCV unique visitor presence counter
two_line_visitor_counter.py     <- Draw two lines and count Line 1 -> Line 2 entries
receptionist_count.py           <- Local OpenCV receptionist counter
requirements.txt                <- Python dependencies
yolov8n.pt                      <- YOLOv8 model weights
receptionist_uniform_ref.png    <- Reference image for receptionist uniform
reception_zone.json             <- Four-point reception desk zone
two_line_counter_lines.json     <- Saved Line 1 and Line 2 visitor counter setup
visitor_entry_zone.json         <- Legacy four-point visitor entrance zone
```

## Local Setup

```bash
pip install -r requirements.txt
streamlit run dual_counter_viewer.py
```

If you run Streamlit through Python, include the `run` command:

```bash
python -m streamlit run dual_counter_viewer.py
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
- Visitor entries are counted when a non-receptionist track crosses the saved
  two lines in the selected entry direction.
- Visitor exits are counted when a non-receptionist track crosses the same two
  lines in the opposite direction.
- Confirmed receptionist IDs are excluded from visitor entry/exit counts.

## Main Metrics

- Receptionists currently at desk
- Peak receptionists at desk
- Confirmed receptionist IDs
- Visitor entries
- Visitor exits
- Frames processed

## Calibration Files

- Edit `reception_zone.json` when the yellow reception zone is not aligned with the desk.
- Replace `receptionist_uniform_ref.png` when the uniform color/reference changes.
- Run `python two_line_visitor_counter.py --redraw` to redraw Line 1 and Line 2
  when the visitor entry/exit lines are not aligned.
