"""
DefectVision Streamlit app.

Image-based industrial defect detection using a supervised SVM model.
"""

import os
import sys

import cv2
import streamlit as st
from PIL import Image


ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.insert(0, SRC_DIR)


st.set_page_config(
    page_title="DefectVision",
    page_icon="DV",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_modules():
    # generate_demo_data may have been removed; provide safe fallbacks if missing.
    try:
        from generate_demo_data import generate_dataset, make_defective_surface, make_good_surface
    except Exception:
        def generate_dataset(*args, **kwargs):
            raise RuntimeError("Demo dataset generation is not available. `generate_demo_data` module missing.")

        def make_defective_surface():
            import numpy as _np
            return _np.full((256, 256, 3), 50, dtype=_np.uint8)

        def make_good_surface():
            import numpy as _np
            return _np.full((256, 256, 3), 200, dtype=_np.uint8)

    from predict import predict_single
    from train import load_model, train_model

    return {
        "generate_dataset": generate_dataset,
        "make_defective_surface": make_defective_surface,
        "make_good_surface": make_good_surface,
        "predict_single": predict_single,
        "load_model": load_model,
        "train_model": train_model,
    }


modules = get_modules()

for key, value in {
    "model_trained": False,
    "train_results": None,
    "history": [],
    "last_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("DefectVision")
st.caption("Image-based industrial defect inspection with supervised SVM")
st.divider()


with st.sidebar:
    st.header("Control Panel")
    st.caption("Supervised SVM trained from labeled good and defective images.")
    st.divider()

    model_bundle, scaler = modules["load_model"]()
    if model_bundle is not None:
        st.session_state.model_trained = True

    if st.session_state.model_trained:
        st.success("SVM model ready")
    else:
        st.error("SVM model not trained")

    st.divider()
    st.subheader("Dataset")

    if st.button("Generate Demo Dataset", width="stretch"):
        with st.spinner("Generating synthetic inspection images..."):
            n_images = modules["generate_dataset"](DATA_DIR, n_good=400, n_defective=400)
        st.success(f"Generated {n_images} images")

    with st.expander("Dataset Layout"):
        st.caption(
            "Place images in:\n\n"
            "data/train/good/\n"
            "data/train/defective/\n"
            "data/test/good/\n"
            "data/test/defective/"
        )
        st.info("Training uses labeled good and defective images. Test folders are optional but recommended.")

    st.divider()
    st.subheader("Training")

    if st.button("Train SVM Model", width="stretch"):
        train_dir = os.path.join(DATA_DIR, "train")
        if not os.path.isdir(train_dir):
            st.error("Create data/train/good and data/train/defective first.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(value, message):
                progress_bar.progress(value)
                status_text.caption(message)

            try:
                results = modules["train_model"](train_dir, progress_callback=update_progress)
                st.session_state.train_results = results
                st.session_state.model_trained = True
                model_bundle, scaler = modules["load_model"]()
                st.success(f"Training complete: {results['accuracy'] * 100:.1f}% train accuracy")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    if st.session_state.history:
        st.divider()
        st.subheader("Session")
        total = len(st.session_state.history)
        defective = sum(1 for item in st.session_state.history if item["label_id"] == 1)
        col_a, col_b = st.columns(2)
        col_a.metric("Inspected", total)
        col_b.metric("Defect Rate", f"{defective / total * 100:.0f}%")
        if st.button("Clear History", width="stretch"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()


model_ready = st.session_state.model_trained
tab_inspect, tab_training, tab_pipeline, tab_history = st.tabs([
    "Inspect Image",
    "Training Results",
    "Pipeline",
    "Inspection History",
])


with tab_inspect:
    col_upload, col_result = st.columns(2, gap="large")

    with col_upload:
        st.subheader("Input Image")
        uploaded = st.file_uploader(
            "Upload part image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )

        demo_img = None
        demo_good, demo_bad = st.columns(2)
        with demo_good:
            if st.button("Good Demo", width="stretch"):
                demo_img = Image.fromarray(
                    cv2.cvtColor(modules["make_good_surface"](), cv2.COLOR_BGR2RGB)
                )
        with demo_bad:
            if st.button("Defective Demo", width="stretch"):
                demo_img = Image.fromarray(
                    cv2.cvtColor(modules["make_defective_surface"](), cv2.COLOR_BGR2RGB)
                )

        source_img = Image.open(uploaded).convert("RGB") if uploaded else demo_img
        if source_img is not None:
            st.image(source_img, caption="Inspection input", width="stretch")

            if not model_ready:
                st.warning("Train the SVM model before running inspection.")
            elif st.button("Run Inspection", width="stretch"):
                with st.spinner("Analyzing image..."):
                    model_bundle, scaler = modules["load_model"]()
                    result = modules["predict_single"](source_img, model_bundle, scaler)
                st.session_state.last_result = result
                st.session_state.history.append(result)
                st.rerun()

    with col_result:
        st.subheader("Inspection Result")
        result = st.session_state.last_result
        if result is None:
            st.info("Upload an image or choose a demo image, then run inspection.")
        else:
            if result["label_id"] == 0:
                st.success("PASS - Good part")
            else:
                st.error("FAIL - Defect detected")

            st.image(result["annotated"], caption="Annotated output", width="stretch")

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            metric_b.metric("Inference", f"{result['inference_ms']:.0f} ms")
            metric_c.metric("Est. FPS", f"{1000 / max(result['inference_ms'], 1):.0f}")

            st.subheader("Class Probability")
            st.caption(f"Good: {result['proba_good'] * 100:.1f}%")
            st.progress(result["proba_good"])
            st.caption(f"Defective: {result['proba_defective'] * 100:.1f}%")
            st.progress(result["proba_defective"])


with tab_training:
    if st.session_state.train_results is None:
        st.info("Train the SVM model from the sidebar to view metrics.")
    else:
        res = st.session_state.train_results
        st.subheader("SVM Performance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{res['accuracy'] * 100:.1f}%")
        col2.metric("F1 Score", f"{res['f1_score']:.3f}")
        col3.metric("Balanced Acc.", f"{res['balanced_accuracy']:.3f}")
        col4.metric("CV F1", f"{res['cv_mean']:.3f} +/- {res['cv_std']:.3f}")

        st.caption(f"Best SVM parameters: {res['best_params']}")

        test_results = res.get("test_results")
        if test_results:
            st.subheader("Held-Out Test")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Accuracy", f"{test_results['accuracy'] * 100:.1f}%")
            t2.metric("F1 Score", f"{test_results['f1_score']:.3f}")
            t3.metric("Balanced Acc.", f"{test_results['balanced_accuracy']:.3f}")
            t4.metric("Samples", test_results["n_samples"])

        col_cm, col_cv = st.columns(2)
        with col_cm:
            if os.path.exists(res["cm_path"]):
                st.image(res["cm_path"], caption="Training confusion matrix", width="stretch")
        with col_cv:
            if os.path.exists(res["cv_path"]):
                st.image(res["cv_path"], caption="Cross-validation scores", width="stretch")

        if test_results and res.get("test_cm_path") and os.path.exists(res["test_cm_path"]):
            st.image(res["test_cm_path"], caption="Held-out test confusion matrix", width="stretch")

        st.subheader("Classification Report")
        st.code(res["report"], language=None)
        if test_results:
            st.subheader("Held-Out Test Report")
            st.code(test_results["report"], language=None)


with tab_pipeline:
    st.subheader("Preprocessing Pipeline")
    result = st.session_state.last_result
    if result is None:
        st.info("Run an image inspection to view the preprocessing stages.")
    else:
        cols = st.columns(4)
        for idx, (name, image) in enumerate(result["stages"].items()):
            with cols[idx % 4]:
                st.image(image, caption=name, width="stretch")


with tab_history:
    st.subheader("Inspection History")
    if not st.session_state.history:
        st.info("No inspections yet.")
    else:
        rows = [{
            "#": idx + 1,
            "Result": item["label"],
            "Confidence": f"{item['confidence'] * 100:.1f}%",
            "P(Good)": f"{item['proba_good'] * 100:.1f}%",
            "P(Defective)": f"{item['proba_defective'] * 100:.1f}%",
            "Inference (ms)": f"{item['inference_ms']:.1f}",
        } for idx, item in enumerate(st.session_state.history)]
        st.dataframe(rows, width="stretch", hide_index=True)

        if len(st.session_state.history) > 1:
            defect_probs = [item["proba_defective"] * 100 for item in st.session_state.history]
            st.line_chart(defect_probs)
