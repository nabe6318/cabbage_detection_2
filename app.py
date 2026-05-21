from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI-Based Cabbage Detection | Tradi-Smart Shinshu",
    layout="wide"
)

# ── Header ──────────────────────────────────────
st.markdown("""
<div style="font-size:0.85rem; color:#555; line-height:1.6; margin-bottom:0.5rem;">
    <span style="color:#c0392b; font-weight:600;">Tradi-Smart Shinshu Program</span><br>
    1<sup>st</sup> day 2026/06/04 &nbsp;&middot;&nbsp; AI-Based Cabbage Detection
</div>
""", unsafe_allow_html=True)

st.title("Cabbage detection demo using an AI model")
st.subheader("Where are the cabbages ready for harvest?")
st.caption(
    "Detect cabbages from UAV imagery using YOLOv8. "
    "Select an image from the gallery or upload your own."
)
st.markdown(
    "<div style='font-size:0.78rem; color:#888; margin-top:-8px;'>"
    "Created by Osamu Watanabe, Weed Lab, Shinshu University."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =========================
# Model path
# =========================
MODEL_PATH = Path("models/best.pt")

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Please place your trained YOLOv8 model file at: models/best.pt")
    st.stop()

model = load_model(str(MODEL_PATH))

# =========================
# Sidebar settings
# =========================
st.sidebar.header("⚙️ Detection settings")

conf_thres = st.sidebar.slider(
    "Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05
)
imgsz = st.sidebar.selectbox(
    "Inference image size", options=[640, 800, 1024, 1280], index=0
)
box_thickness = st.sidebar.slider(
    "Box line thickness", min_value=1, max_value=5, value=2, step=1
)
show_label = st.sidebar.checkbox("Show label & confidence", value=False)

# =========================
# Gallery image folder
# =========================
GALLERY_DIR = Path("hex_images_jpg")

def get_gallery_images() -> list[Path]:
    if not GALLERY_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        [p for p in GALLERY_DIR.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem
    )

gallery_images = get_gallery_images()

# =========================
# Session state
# =========================
if "selected_gallery_path" not in st.session_state:
    st.session_state["selected_gallery_path"] = None

# =========================
# Gallery section
# =========================
if gallery_images:
    st.markdown("### 📂 Image Gallery (click to select)")
    st.caption(f"{GALLERY_DIR} — {len(gallery_images)} images available. Scroll to see all.")

    THUMB_COLS = 7
    THUMB_SIZE = (160, 160)
    rows = [gallery_images[i:i + THUMB_COLS]
            for i in range(0, len(gallery_images), THUMB_COLS)]

    with st.container(height=380, border=True):
        for row in rows:
            cols = st.columns(len(row))
            for col, img_path in zip(cols, row):
                with col:
                    try:
                        thumb = Image.open(img_path).convert("RGB")
                        thumb.thumbnail(THUMB_SIZE, Image.LANCZOS)
                    except Exception:
                        st.warning(img_path.name)
                        continue
                    is_selected = (
                        st.session_state["selected_gallery_path"] == str(img_path)
                    )
                    st.image(thumb, use_container_width=True)
                    btn_label = f"{'✅ ' if is_selected else ''}{img_path.stem}"
                    if st.button(btn_label, key=f"btn_{img_path.stem}",
                                 use_container_width=True):
                        st.session_state["selected_gallery_path"] = str(img_path)
                        st.rerun()

    st.markdown("---")
else:
    st.info(
        f"Gallery folder `{GALLERY_DIR}` not found. "
        "Create the folder and add images to display thumbnails here."
    )

# =========================
# Image source selection
# =========================
st.markdown("### 🖼️ Select image for detection")

tab_gallery, tab_upload = st.tabs(["📂 Select from gallery", "📤 Upload image"])

selected_image: Image.Image | None = None
selected_name: str = ""

with tab_gallery:
    if st.session_state["selected_gallery_path"]:
        p = Path(st.session_state["selected_gallery_path"])
        st.success(f"Selected: **{p.name}**")
        try:
            selected_image = Image.open(p).convert("RGB")
            selected_name = p.stem
            preview_col, _ = st.columns([1, 2])
            with preview_col:
                st.image(selected_image, caption=p.name, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image: {e}")
            selected_image = None
    else:
        st.info("Click an image in the gallery above to select it.")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file).convert("RGB")
            selected_name = Path(uploaded_file.name).stem
            st.session_state["selected_gallery_path"] = None
        except Exception as e:
            st.error(f"Could not read the uploaded image: {e}")

# =========================
# Guard
# =========================
if selected_image is None:
    st.info("Select an image from the gallery or upload one to continue.")
    st.stop()

# =========================
# Run detection button
# =========================
st.markdown("---")
run_detection = st.button("▶️ Run detection", type="primary", use_container_width=True)

if not run_detection:
    st.stop()

# =========================
# Inference
# =========================
image_np = np.array(selected_image)
height, width = image_np.shape[:2]
st.write(f"Image size: {width} × {height} px")

with st.spinner("Detecting with YOLO..."):
    results = model.predict(
        source=image_np,
        imgsz=imgsz,
        conf=conf_thres,
        verbose=False,
    )

result = results[0]

# =========================
# Extract detections
# =========================
detections = []
annotated = image_np.copy()

if result.boxes is not None and len(result.boxes) > 0:
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs      = result.boxes.conf.cpu().numpy()
    cls_ids    = result.boxes.cls.cpu().numpy().astype(int)

    for i, (box, conf, cls_id) in enumerate(
        zip(boxes_xyxy, confs, cls_ids), start=1
    ):
        x1, y1, x2, y2 = box
        box_width  = x2 - x1
        box_height = y2 - y1
        box_area   = box_width * box_height
        center_x   = (x1 + x2) / 2
        center_y   = (y1 + y2) / 2
        class_name = model.names.get(cls_id, str(cls_id))

        detections.append({
            "id": i,
            "class": class_name,
            "confidence": round(float(conf), 4),
            "x1_px": round(float(x1), 1),
            "y1_px": round(float(y1), 1),
            "x2_px": round(float(x2), 1),
            "y2_px": round(float(y2), 1),
            "center_x_px": round(float(center_x), 1),
            "center_y_px": round(float(center_y), 1),
            "box_width_px": round(float(box_width), 1),
            "box_height_px": round(float(box_height), 1),
            "box_area_px2": round(float(box_area), 1),
        })

        x1_i = max(0, min(int(x1), width - 1))
        y1_i = max(0, min(int(y1), height - 1))
        x2_i = max(0, min(int(x2), width - 1))
        y2_i = max(0, min(int(y2), height - 1))

        cv2.rectangle(
            annotated, (x1_i, y1_i), (x2_i, y2_i),
            color=(255, 0, 0), thickness=box_thickness,
        )

        if show_label:
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated, label, (x1_i, max(y1_i - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
            )

df = pd.DataFrame(detections)

# =========================
# Display results
# =========================
st.subheader("Detection results")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Original image")
    st.image(image_np, use_container_width=True)
with col2:
    st.markdown("#### Detection result")
    st.image(annotated, use_container_width=True)

st.markdown("---")
st.metric("Cabbages detected", len(df))

if len(df) > 0:
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Download detection CSV",
        data=csv,
        file_name=f"cabbage_detection_{selected_name}.csv",
        mime="text/csv",
    )

    # =========================================================
    # Box size analysis
    # =========================================================
    st.markdown("---")
    st.subheader("📏 Box size analysis")

    import matplotlib.pyplot as plt

    areas = df["box_area_px2"]

    # ── Outlier filter (IQR method) ───────────────────────────
    q1        = float(areas.quantile(0.25))
    q3        = float(areas.quantile(0.75))
    iqr       = q3 - q1
    iqr_upper = q3 + 1.5 * iqr   # standard upper-outlier boundary

    st.markdown("#### 🔍 Outlier filter — upper bound")
    st.caption(
        "Abnormally large boxes (merged detections, background noise, etc.) "
        "can distort the histogram. "
        "The IQR threshold (Q3 + 1.5 × IQR) is calculated automatically. "
        "Adjust the slider if needed."
    )

    col_sl, col_info = st.columns([3, 1])
    with col_sl:
        max_area = st.slider(
            "Maximum box area (px²) — boxes above this are excluded as outliers",
            min_value=int(areas.min()),
            max_value=int(areas.max()),
            value=min(int(iqr_upper), int(areas.max())),
            step=100,
        )
    with col_info:
        st.markdown(
            f"""
            <div style='background:#fff8e1;border:1px solid #ffb300;
                        border-radius:6px;padding:10px 12px;font-size:0.8rem;'>
            <b>Auto IQR</b><br>
            Q1 = {q1:,.0f}<br>
            Q3 = {q3:,.0f}<br>
            IQR = {iqr:,.0f}<br>
            <b>Limit = {iqr_upper:,.0f} px²</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Split into normal / outlier
    df_filtered  = df[df["box_area_px2"] <= max_area].copy()
    df_outliers  = df[df["box_area_px2"] >  max_area].copy()
    n_outliers   = len(df_outliers)
    areas_f      = df_filtered["box_area_px2"]

    if n_outliers > 0:
        st.warning(
            f"⚠️ **{n_outliers} box(es) excluded** "
            f"(area > {max_area:,} px²). "
            "Statistics and histogram use the filtered data only."
        )
    else:
        st.success("✅ No outliers detected at the current threshold.")

    # ── Statistics ─────────────────────────────────────────────
    st.markdown("#### 📊 Box area statistics (filtered, px²)")

    if len(areas_f) == 0:
        st.warning("No boxes remain. Try raising the maximum area threshold.")
        st.stop()

    stats = areas_f.describe()
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Count",  int(stats["count"]))
    s2.metric("Mean",   f"{stats['mean']:.0f}")
    s3.metric("Median", f"{areas_f.median():.0f}")
    s4.metric("Min",    f"{stats['min']:.0f}")
    s5.metric("Max",    f"{stats['max']:.0f}")

    # ── Histogram ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(areas_f, bins=30, alpha=0.85, color="#4CAF50", edgecolor="white",
            label="Detected boxes (filtered)")
    ax.axvline(float(areas_f.median()), color="#E53935", linestyle="--",
               linewidth=1.6, label=f"Median: {areas_f.median():.0f} px²")
    ax.axvline(float(areas_f.mean()), color="#1565C0", linestyle=":",
               linewidth=1.6, label=f"Mean: {areas_f.mean():.0f} px²")
    ax.set_xlabel("Box area (px²)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Box area distribution  "
        f"(n={len(areas_f)}"
        + (f", {n_outliers} outlier(s) removed" if n_outliers > 0 else "") + ")"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Harvest-ready filter (lower bound) ─────────────────────
    st.markdown("#### 🥬 Harvest-ready filter — minimum size")
    st.caption(
        "Boxes smaller than the threshold are treated as immature cabbages."
    )

    min_area = st.slider(
        "Minimum box area (px²) — boxes below this are considered immature",
        min_value=int(areas_f.min()),
        max_value=int(areas_f.max()),
        value=int(areas_f.quantile(0.25)),
        step=100,
    )

    harvest_ready = df_filtered[df_filtered["box_area_px2"] >= min_area]
    immature      = df_filtered[df_filtered["box_area_px2"] <  min_area]

    hr1, hr2, hr3 = st.columns(3)
    hr1.metric("🥬 Harvest-ready", len(harvest_ready))
    hr2.metric("🌱 Immature / small", len(immature))
    hr3.metric(
        "Harvest ratio",
        f"{len(harvest_ready) / len(df_filtered) * 100:.1f}%",
    )

    # ── Detail table ───────────────────────────────────────────
    with st.expander("View all box details (including excluded outliers)"):
        df_display = df_filtered.copy()
        df_display["status"] = df_display["box_area_px2"].apply(
            lambda a: "harvest-ready" if a >= min_area else "immature"
        )
        df_display["outlier_excluded"] = False

        if n_outliers > 0:
            df_out = df_outliers.copy()
            df_out["status"] = "outlier (excluded)"
            df_out["outlier_excluded"] = True
            df_display = pd.concat([df_display, df_out]).sort_values("id")

        st.dataframe(df_display, use_container_width=True)

    csv_filtered = df_filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Download filtered box CSV",
        data=csv_filtered,
        file_name=f"cabbage_boxes_filtered_{selected_name}.csv",
        mime="text/csv",
    )

else:
    st.warning("No cabbages detected. Try lowering the confidence threshold.")
