import json
import os
import tempfile

import streamlit as st
from counter_core import analyze_video


def main():


    st.set_page_config(page_title="Receptionist + Visitor Counters", layout="wide")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
    REFERENCE_PATH = os.path.join(BASE_DIR, "receptionist_uniform_ref.png")
    ZONE_PATH = os.path.join(BASE_DIR, "reception_zone.json")
    TWO_LINE_PATH = os.path.join(BASE_DIR, "two_line_counter_lines.json")

    ref_w = 1920
    ref_h = 1080
    yolo_conf = 0.50
    yolo_iou = 0.40
    match_threshold = 0.55
    min_red = 20
    min_overlap = 0.30
    two_line_path = TWO_LINE_PATH
    entry_order_option = "1-2"
    analyze_every = 1

    st.title("Receptionist and two-line visitor counter")
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

    with st.sidebar:
        st.header("Video")
        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
        )

    st.markdown("---")

    if st.button("Run counters", type="primary"):
        temp_video_path = None
        video_path = ""
        if uploaded_video is None:
            st.error("Upload a video first.")
            st.stop()

        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.getbuffer())
            temp_video_path = tmp.name
        video_path = temp_video_path

        if not os.path.isfile(video_path):
            st.error(f"Video not found: `{video_path}`")
            if temp_video_path is not None:
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass
            st.stop()
        if not os.path.isfile(two_line_path):
            st.error(
                f"Two-line file not found: `{two_line_path}`. "
                "Run `python visitor_counter.py --redraw` first."
            )
            if temp_video_path is not None:
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass
            st.stop()

        progress_bar = st.progress(0, text="Starting analysis...")
        status = st.empty()
        frame_preview = st.empty()

        def update_progress(frame_num, total_frames):
            if total_frames > 0:
                progress_bar.progress(
                    min(frame_num / total_frames, 1.0),
                    text=f"Frame {frame_num} / {total_frames}",
                )
            else:
                progress_bar.progress(0, text=f"Frame {frame_num}")

        def update_frame(frame, frame_num, total_frames):
            update_progress(frame_num, total_frames)
            frame_preview.image(frame, channels="BGR", width="stretch")

        try:
            result = analyze_video(
                video_path=video_path,
                model_path=MODEL_PATH,
                reference_path=REFERENCE_PATH,
                zone_path=ZONE_PATH,
                two_line_path=two_line_path,
                ref_w=ref_w,
                ref_h=ref_h,
                entry_order_option=entry_order_option,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                match_threshold=match_threshold,
                min_red=min_red,
                min_overlap=min_overlap,
                analyze_every=analyze_every,
                on_progress=update_progress,
                on_frame=update_frame,
                preview_every=5,
            )
        except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            st.error(str(exc))
            if temp_video_path is not None:
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass
            st.stop()

        if temp_video_path is not None:
            try:
                os.remove(temp_video_path)
            except OSError:
                pass

        progress_bar.progress(1.0, text="Done")
        status.write("**Analysis complete.**")

        st.success(
            f"Finished. Peak receptionists at desk: {result.peak_receptionists}. "
            f"Visitor entries: {result.visitor_entries}, exits: {result.visitor_exits}."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Frames processed", result.frames_processed)
        with c2:
            st.metric("Peak receptionists at desk", result.peak_receptionists)
        with c3:
            st.metric("Confirmed receptionist IDs", result.confirmed_receptionist_ids)
        with c4:
            st.metric("Visitor entries", result.visitor_entries)
        with c5:
            st.metric("Visitor exits", result.visitor_exits)
    st.markdown("---")
    st.caption(
        "`reception_zone.json` controls the four-point reception desk zone. "
        "`receptionist_uniform_ref.png` controls the uniform color match. "
        "`two_line_counter_lines.json` controls visitor entry/exit lines."
    )

if __name__ == "__main__":
    main()

   
