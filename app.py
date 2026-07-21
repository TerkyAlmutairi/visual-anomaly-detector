"""
app.py

Streamlit front end. Upload a handful of "normal" reference images (photos
of what a surface/object/scene looks like when everything's fine) and one
query image — the app flags anomalous regions with a heatmap, using a
pretrained CNN backbone and zero labeled defect data.

Run locally:  streamlit run app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import AnomalyDetector  # noqa: E402

st.set_page_config(page_title="PatchSense", page_icon="🔍", layout="centered")

st.title("🔍 PatchSense")
st.caption(
    "PatchSense is a few-shot anomaly detector inspired by PatchCore (Roth et al., 2022). "
    "Upload photos of what 'normal' looks like, then upload a photo to check — "
    "no labeled defect data needed, and no fixed category of object."
)

st.markdown(
    "**How it works:** a pretrained CNN extracts local texture/structure features "
    "from your reference images, building a 'memory' of what normal looks like. "
    "Your query image is then compared patch-by-patch against that memory — regions "
    "unlike anything in the references get flagged, with a heatmap showing exactly where."
)

if "detector" not in st.session_state:
    with st.spinner("Loading pretrained model (first run only, ~15s)..."):
        st.session_state["detector"] = AnomalyDetector(pretrained=True)

st.subheader("1. Reference images (what 'normal' looks like)")
ref_files = st.file_uploader(
    "Upload 3-5 images of a defect-free example (e.g. a clean wall, an undamaged part, a normal scene)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

st.subheader("2. Query image (what you want checked)")
query_file = st.file_uploader("Upload the image to inspect", type=["jpg", "jpeg", "png"])

run_button = st.button("Run detection", type="primary", use_container_width=True)

if run_button:
    if not ref_files or len(ref_files) < 1:
        st.error("Upload at least one reference image (3+ recommended for reliable calibration).")
    elif not query_file:
        st.error("Upload a query image to check.")
    else:
        ref_images = [Image.open(f) for f in ref_files]
        query_image = Image.open(query_file)

        with st.spinner("Extracting features and comparing against reference patches..."):
            result = st.session_state["detector"].detect(ref_images, query_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(query_image, caption="Original", use_container_width=True)
        with col2:
            st.image(result.heatmap_overlay, caption="Anomaly heatmap", use_container_width=True)

        if result.is_anomalous:
            st.error(f"⚠️ Anomaly detected — score: {result.anomaly_score} (threshold: 100)")
        else:
            st.success(f"✅ No anomaly detected — score: {result.anomaly_score} (threshold: 100)")

        with st.expander("Scoring details"):
            st.write(f"Raw max patch distance: {result.raw_max_score:.3f}")
            st.write(f"Calibration noise floor (from reference images): {result.calibration_max:.3f}")
            st.write(
                "The score is the raw distance scaled against the natural variation "
                "seen *among the reference images themselves* — so 100 means 'as "
                "different as your normal images are from each other,' not an arbitrary cutoff."
            )
            if len(ref_images) < 2:
                st.warning(
                    "Only 1 reference image provided — calibration is using a conservative "
                    "default rather than measured noise floor. Add 2+ references for a properly "
                    "calibrated score."
                )
else:
    st.info("Upload reference and query images above, then click **Run detection**.")
