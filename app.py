"""
app.py
------
Streamlit UI for Industrial Defect Detection System.
Supports both Classic (R-CNN) and R-CNN (Deep Learning) models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import cv2
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import tempfile

st.set_page_config(
    page_title="DefectVision",
    page_icon="DV",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_modules():
    from train import train_model, load_model
    from predict import predict_single, make_pipeline_figure
    from generate_demo_data import generate_dataset
    from video_inference import process_video, read_video_bytes
    return train_model, load_model, predict_single, make_pipeline_figure, \
           generate_dataset, process_video, read_video_bytes


train_model, load_model, predict_single, make_pipeline_figure, \
    generate_dataset, process_video, read_video_bytes = get_modules()

for k, v in {"model_trained": False, 
             "train_results": None, 
             "history": [], "video_result": None, "last_result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


st.title("DefectVision")
st.caption("Industrial defect detection")
st.divider()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Control Panel")

    st.caption("OneClassSVM anomaly detector with hand-crafted features (fast, lightweight)")

    st.divider()

    # --- Load existing models ---
    clf, scaler = load_model()
    if clf is not None:
        st.session_state.model_trained = True

    # --- Model status ---
    if st.session_state.model_trained:
        st.success("R-CNN Model Ready")
    else:
        st.error("R-CNN Model Not Trained")

    st.divider()
    st.subheader("Dataset")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    if st.button("Generate Demo Dataset", width="stretch"):
        with st.spinner("Generating synthetic images..."):
            n = generate_dataset(DATA_DIR, n_good=400, n_defective=400)
        st.success(f"Generated {n} images")

    with st.expander("Use Real Dataset"):
        st.caption("Place images in:\n```\ndata/train/good/\ndata/train/defective/\n```\nThen click Train Model.")
        st.warning("Retrain on images similar to your real uploads. Otherwise the saved model may keep predicting one class for very different inputs.")
        st.info("Training uses only `data/train/good` and `data/train/defective`.")

    st.divider()
    st.subheader("Training")

    # R-CNN Training
    if st.button("Train R-CNN Model", width="stretch"):
        train_dir = os.path.join(DATA_DIR, "train")
        if not os.path.exists(train_dir):
            st.error("Run 'Generate Demo Dataset' first.")
        else:
            progress_bar = st.progress(0)
            status_text_el = st.empty()

            def update(val, msg):
                progress_bar.progress(val)
                status_text_el.caption(msg)

            try:
                results = train_model(train_dir, progress_callback=update)
                st.session_state.train_results = results
                st.session_state.model_trained = True
                clf, scaler = load_model()
                st.success(f"Accuracy: {results['accuracy']*100:.1f}%")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.history:
        st.divider()
        st.subheader("Session Stats")
        total = len(st.session_state.history)
        defective = sum(1 for h in st.session_state.history if h["label_id"] == 1)
        col1, col2 = st.columns(2)
        col1.metric("Inspected", total)
        col2.metric("Defect Rate", f"{defective/total*100:.0f}%")
        if st.button("Clear History", width="stretch"):
            st.session_state.history = []
            st.rerun()


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Inspect",
    "Video Analysis",
    "Training Results",
    "Pipeline Viewer",
    "Batch History",
])

# Determine which model is active
model_ready = st.session_state.model_trained


# ── Tab 1: Inspect ───────────────────────────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns(2, gap="large")

    with col_upload:
        st.subheader("Input Image")
        uploaded = st.file_uploader("Upload part image", type=["png", "jpg", "jpeg", "bmp"],
                                    label_visibility="collapsed")
        st.caption("Or use a demo image:")
        d1, d2 = st.columns(2)
        demo_img = None
        with d1:
            if st.button("Good Part", width="stretch"):
                from generate_demo_data import make_good_surface
                demo_img = Image.fromarray(cv2.cvtColor(make_good_surface(), cv2.COLOR_BGR2RGB))
        with d2:
            if st.button("Defective Part", width="stretch"):
                from generate_demo_data import make_defective_surface
                demo_img = Image.fromarray(cv2.cvtColor(make_defective_surface(), cv2.COLOR_BGR2RGB))

        source_img = Image.open(uploaded).convert("RGB") if uploaded else demo_img
        if source_img:
            st.image(source_img, caption="Input Image", width="stretch")
            if not model_ready:
                st.warning("Train the R-CNN model first using the sidebar.")
            elif st.button("Run Inspection", width="stretch"):
                with st.spinner("Analyzing..."):
                    clf, scaler = load_model()
                    result = predict_single(source_img, clf, scaler)
                st.session_state.last_result = result
                st.session_state.history.append(result)
                st.rerun()

    with col_result:
        st.subheader("Inspection Result")
        r = st.session_state.last_result
        if r:
            if r["label"] == "Good":
                st.success("PASS - Good Part")
            else:
                st.error("FAIL - Defect Detected")

            st.image(r["annotated"], caption="Annotated Output", width="stretch")

            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{r['confidence']*100:.1f}%")
            m2.metric("Inference", f"{r['inference_ms']:.0f} ms")
            m3.metric("Est. FPS", f"{1000/max(r['inference_ms'],1):.0f}")

            st.subheader("Confidence Distribution")
            fig = go.Figure(go.Bar(
                x=["Good", "Defective"],
                y=[r["proba_good"]*100, r["proba_defective"]*100],
                marker_color=["#2ecc71", "#e74c3c"],
                text=[f"{r['proba_good']*100:.1f}%", f"{r['proba_defective']*100:.1f}%"],
                textposition="outside",
            ))
            fig.update_layout(yaxis=dict(range=[0, 115], ticksuffix="%"),
                              height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Upload an image or use a demo button, then run inspection.")


# ── Tab 2: Video Analysis ────────────────────────────────────────────────────
with tab2:
    st.subheader("Video Defect Analysis")

    if not model_ready:
        st.warning("Train the R-CNN model first using the sidebar.")
    else:
        col_vid, col_cfg = st.columns([2, 1], gap="large")

        with col_vid:
            uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"],
                                              label_visibility="collapsed", key="video_uploader")
            if uploaded_video:
                st.video(uploaded_video)

        with col_cfg:
            st.subheader("Settings")
            frame_skip = st.slider("Analyze every N frames", 1, 10, 3,
                                   help="Higher = faster, lower temporal resolution")
            st.info(f"Every **{frame_skip}** frame(s) analyzed")

        if uploaded_video and st.button("Analyze Video", width="stretch", key="run_video"):
            suffix = "." + uploaded_video.name.split(".")[-1]
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_in.write(uploaded_video.read())
            tmp_in.close()

            pb = st.progress(0)
            se = st.empty()

            def vid_progress(val, msg):
                pb.progress(val)
                se.caption(msg)

            try:
                with st.spinner(""):
                    clf, scaler = load_model()
                    vr = process_video(tmp_in.name, clf, scaler,
                                       frame_skip=frame_skip,
                                       progress_callback=vid_progress)
                st.session_state.video_result = vr
                os.unlink(tmp_in.name)
                st.rerun()
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                os.unlink(tmp_in.name)

        vr = st.session_state.video_result
        if vr:
            st.divider()
            st.subheader("Analysis Results")
            defect_pct = vr["defect_rate"] * 100
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duration", f"{vr['duration_s']:.1f}s")
            m2.metric("Frames Analyzed", vr["processed_frames"])
            m3.metric("Defect Rate", f"{defect_pct:.1f}%")
            m4.metric("FPS", f"{vr['fps']:.0f}")

            st.subheader("Defect Timeline")
            fr = vr["frame_results"]
            timestamps = [row["timestamp_s"] for row in fr]
            prob_defect = [row["proba_defective"]*100 for row in fr]
            marker_colors = ["#e74c3c" if row["label"] == "Defective" else "#2ecc71" for row in fr]

            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=timestamps, y=prob_defect, mode="lines",
                line=dict(color="#3498db", width=1.5), name="P(Defective)",
                fill="tozeroy", fillcolor="rgba(52,152,219,0.06)",
            ))
            fig_tl.add_trace(go.Scatter(
                x=timestamps, y=prob_defect, mode="markers",
                marker=dict(color=marker_colors, size=5), showlegend=False,
            ))
            fig_tl.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1,
                             annotation_text="Decision boundary (50%)",
                             annotation_font=dict(color="gray", size=10))
            fig_tl.update_layout(
                xaxis=dict(title="Time (seconds)", ticksuffix="s"),
                yaxis=dict(title="P(Defective) %", ticksuffix="%", range=[0, 108]),
                height=340, margin=dict(l=10, r=10, t=16, b=10),
            )
            st.plotly_chart(fig_tl, width="stretch")

            st.download_button(
                label="Download Annotated Video",
                data=read_video_bytes(vr["output_path"]),
                file_name="defect_annotated.mp4",
                mime="video/mp4",
                width="stretch",
            )

            with st.expander("View per-frame results table"):
                df_frames = pd.DataFrame([{
                    "Frame": row["frame_idx"],
                    "Time (s)": f"{row['timestamp_s']:.2f}",
                    "Result": row["label"],
                    "P(Defective)": f"{row['proba_defective']*100:.1f}%",
                    "Confidence": f"{row['confidence']*100:.1f}%",
                } for row in fr])
                st.dataframe(df_frames, width="stretch", hide_index=True)


# ── Tab 3: Training Results ──────────────────────────────────────────────────
with tab3:
    if not st.session_state.model_trained or st.session_state.train_results is None:
        st.info("Train the model from the sidebar to view results here.")
    else:
        res = st.session_state.train_results
        st.subheader("R-CNN Model Performance")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{res['accuracy']*100:.1f}%")
        c2.metric("F1 Score", f"{res['f1_score']:.3f}")
        c3.metric("CV Mean F1", f"{res['cv_mean']:.3f} ± {res['cv_std']:.3f}")
        c4.metric("Training Samples", res["n_samples"])

        test_results = res.get("test_results")
        if test_results:
            st.subheader("Held-Out Test Performance")
            t1, t2, t3 = st.columns(3)
            t1.metric("Test Accuracy", f"{test_results['accuracy']*100:.1f}%")
            t2.metric("Test F1 Score", f"{test_results['f1_score']:.3f}")
            t3.metric("Test Samples", test_results["n_samples"])

        col_cm, col_cv = st.columns(2)
        with col_cm:
            st.subheader("Confusion Matrix")
            if os.path.exists(res["cm_path"]):
                st.image(res["cm_path"], width="stretch")
        with col_cv:
            st.subheader("Cross-Validation Scores")
            if os.path.exists(res["cv_path"]):
                st.image(res["cv_path"], width="stretch")

        if test_results and res.get("test_cm_path") and os.path.exists(res["test_cm_path"]):
            st.subheader("Held-Out Test Confusion Matrix")
            st.image(res["test_cm_path"], width="stretch")

        st.subheader("Classification Report")
        st.code(res["report"], language=None)
        if test_results:
            st.subheader("Held-Out Test Report")
            st.code(test_results["report"], language=None)

        st.subheader("Dataset Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=["Good", "Defective"],
            values=[res["n_good"], res["n_defective"]],
            hole=0.5,
            marker=dict(colors=["#2ecc71", "#e74c3c"]),
        ))
        fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, width="stretch")


# ── Tab 4: Pipeline Viewer ───────────────────────────────────────────────────
with tab4:
    st.subheader("Pipeline Visualization")
    if st.session_state.last_result:
        r = st.session_state.last_result
        # R-CNN result
        fig_pl = make_pipeline_figure(r["stages"], figsize=(14, 4))
        st.pyplot(fig_pl, width="stretch")

        st.subheader("Stage Descriptions")
        stage_info = {
            "Original": "Raw input resized to 224x224.",
            "Object Focus": "Automatically crops around the main object to reduce background mismatch.",
            "Grayscale": "Color removed because defects are mostly structural rather than color-based.",
            "Contrast Normalize": "CLAHE boosts local contrast so lighting changes do not dominate the prediction.",
            "Gaussian Blur": "5×5 smoothing removes micro-texture noise before thresholding and gradients.",
            "Sobel Edges": "Gradient magnitude highlights sharp boundaries that often correspond to defects.",
            "Adaptive Threshold": "Locally thresholds reflective or unevenly lit surfaces more reliably.",
            "Morphological Clean": "Closing and opening connect broken regions and suppress small noise blobs.",
        }
        cols = st.columns(3)
        for i, (stage, desc) in enumerate(stage_info.items()):
            with cols[i % 3]:
                st.markdown(f"**{stage}**")
                st.caption(desc)
    else:
        st.info("Run an inspection in the Inspect tab to visualize the pipeline.")


# ── Tab 5: Batch History ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Inspection History")
    if not st.session_state.history:
        st.info("No inspections yet. Run inspections in the Inspect tab.")
    else:
        rows = [{
            "#": i + 1,
            "Result": h["label"],
            "Confidence": f"{h['confidence']*100:.1f}%",
            "P(Good)": f"{h['proba_good']*100:.1f}%",
            "P(Defective)": f"{h['proba_defective']*100:.1f}%",
            "Inference (ms)": f"{h['inference_ms']:.1f}",
        } for i, h in enumerate(st.session_state.history)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if len(st.session_state.history) >= 2:
            st.subheader("Defect Confidence Trend")
            conf_def = [h["proba_defective"]*100 for h in st.session_state.history]
            dot_cols = ["#e74c3c" if h["label"] == "Defective" else "#2ecc71"
                        for h in st.session_state.history]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                y=conf_def, mode="lines+markers",
                line=dict(color="#3498db", width=2),
                marker=dict(color=dot_cols, size=10),
                name="P(Defective)",
            ))
            fig_hist.add_hline(y=50, line_dash="dash", line_color="gray",
                               annotation_text="Decision boundary", annotation_font_color="gray")
            fig_hist.update_layout(
                yaxis=dict(range=[0, 105], ticksuffix="%", title="P(Defective) %"),
                xaxis=dict(title="Inspection #"),
                height=300, margin=dict(l=10, r=10, t=10, b=40), showlegend=False,
            )
            st.plotly_chart(fig_hist, width="stretch")
