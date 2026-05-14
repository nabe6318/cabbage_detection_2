# app.py
"""
Cabbage Detection App — Nobeyama
区画タイル分割・ヒートマップ版

Folder structure:
cabbage_detection_1/
├─ app.py
├─ requirements.txt
├─ ALL_area.jpg
├─ models/
│  └─ best.pt
└─ clipped/
   ├─ area_01.jpg / area_01.tif など
   ├─ area_02.jpg / area_02.tif など
   └─ ...
"""

import os
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Cabbage Detection — Nobeyama",
    layout="wide",
    page_icon="🥬",
)

st.title("🥬 Cabbage Detection App — Nobeyama")
st.caption(
    "Detect cabbages from UAV orthomosaic imagery using YOLOv8. "
    "Each area is split into tiles; black NoData tiles are automatically skipped."
)


# =========================
# Paths for Streamlit Cloud / local relative structure
# =========================
APP_DIR = Path(__file__).parent

FIELD_MAP_PATH = APP_DIR / "ALL_area.jpg"
DEFAULT_MODEL_PATH = APP_DIR / "models" / "best.pt"
DEFAULT_IMAGE_DIR = APP_DIR / "clipped"


# =========================
# Field map
# =========================
st.subheader("Field map")

if FIELD_MAP_PATH.exists():
    col_map, col_empty = st.columns([1, 1])
    with col_map:
        st.image(
            str(FIELD_MAP_PATH),
            caption="Nobeyama cabbage field — 20 areas",
            use_container_width=True,
        )
else:
    st.info(
        "Field map image was not found. "
        "If needed, place ALL_area.jpg in the same folder as app.py."
    )


# =========================
# Sidebar settings
# =========================
st.sidebar.header("⚙️ Settings")

model_path = st.sidebar.text_input(
    "YOLOv8 model path",
    value=str(DEFAULT_MODEL_PATH),
)

image_dir = st.sidebar.text_input(
    "Clipped image folder",
    value=str(DEFAULT_IMAGE_DIR),
)

TILE_W = st.sidebar.number_input(
    "Tile width (px)",
    min_value=100,
    max_value=3000,
    value=500,
    step=10,
)

TILE_H = st.sidebar.number_input(
    "Tile height (px)",
    min_value=100,
    max_value=3000,
    value=430,
    step=10,
)

conf_thres = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

black_threshold = st.sidebar.slider(
    "Black area skip threshold (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Skip tiles with black pixels above this ratio.",
)

imgsz = st.sidebar.selectbox(
    "Inference image size",
    options=[640, 800, 1024, 1280],
    index=0,
)

box_thickness = st.sidebar.slider(
    "Box line thickness",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
)


# =========================
# Load model
# =========================
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


model_path_obj = Path(model_path)

if not model_path_obj.exists():
    st.error(f"Model not found: {model_path_obj}")
    st.info(
        "Please place your trained YOLOv8 model file at: models/best.pt"
    )
    st.stop()

model = load_model(str(model_path_obj))


# =========================
# Image file list
# =========================
def find_images(directory: str):
    """
    Find image files in the specified folder and sort by number in filename.
    """
    patterns = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
    files = []

    for pat in patterns:
        files.extend(glob.glob(os.path.join(directory, pat)))

    def sort_key(f):
        name = Path(f).stem
        nums = "".join(c for c in name if c.isdigit())
        return int(nums) if nums else 0

    return sorted(files, key=sort_key)


image_dir_obj = Path(image_dir)

if not image_dir_obj.is_dir():
    st.error(f"Folder not found: {image_dir_obj}")
    st.info(
        "Please place clipped area images in the clipped folder."
    )
    st.stop()

image_files = find_images(str(image_dir_obj))

if not image_files:
    st.error(f"No image files found in: {image_dir_obj}")
    st.stop()

st.sidebar.markdown(f"**Images found: {len(image_files)}**")


# =========================
# Tile split & detection
# =========================
def split_and_detect(
    img_path,
    model,
    tile_w,
    tile_h,
    conf,
    imgsz,
    black_thresh,
    box_thick,
):
    """
    Split image into tiles, skip black NoData areas, and run YOLO inference.

    Returns:
        annotated: annotated RGB image
        tile_results: tile-level detection results
        grid_shape: rows and columns
        box_details: individual bounding box information
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    cols = max(1, w // tile_w)
    rows = max(1, h // tile_h)

    # Include remaining edges if large enough
    if w % tile_w > 50:
        cols += 1
    if h % tile_h > 50:
        rows += 1

    annotated = arr.copy()
    tile_results = []
    box_details = []

    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h
            x1 = c * tile_w
            y2 = min(y1 + tile_h, h)
            x2 = min(x1 + tile_w, w)

            tile = arr[y1:y2, x1:x2]

            if tile.size == 0:
                continue

            # Detect black NoData pixels
            black_mask = np.all(tile < 10, axis=2)
            black_ratio = black_mask.sum() / black_mask.size * 100

            if black_ratio >= black_thresh:
                tile_results.append({
                    "row": r,
                    "col": c,
                    "count": 0,
                    "skipped": True,
                    "black_ratio": round(float(black_ratio), 1),
                })
                continue

            # YOLO inference
            results = model.predict(
                source=tile,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
            )

            result = results[0]

            det_count = 0

            if result.boxes is not None and len(result.boxes) > 0:
                det_count = len(result.boxes)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, cf in zip(boxes_xyxy, confs):
                    bx1, by1, bx2, by2 = map(int, box)

                    # Clip coordinates inside tile
                    bx1 = max(0, min(bx1, tile.shape[1] - 1))
                    bx2 = max(0, min(bx2, tile.shape[1] - 1))
                    by1 = max(0, min(by1, tile.shape[0] - 1))
                    by2 = max(0, min(by2, tile.shape[0] - 1))

                    bw = bx2 - bx1
                    bh = by2 - by1
                    area = bw * bh

                    box_details.append({
                        "tile_row": r,
                        "tile_col": c,
                        "x1_px": x1 + bx1,
                        "y1_px": y1 + by1,
                        "x2_px": x1 + bx2,
                        "y2_px": y1 + by2,
                        "box_w": bw,
                        "box_h": bh,
                        "box_area": area,
                        "confidence": round(float(cf), 4),
                    })

                    # Tile coordinates to whole-image coordinates
                    cv2.rectangle(
                        annotated,
                        (x1 + bx1, y1 + by1),
                        (x1 + bx2, y1 + by2),
                        color=(255, 0, 0),
                        thickness=box_thick,
                    )

            tile_results.append({
                "row": r,
                "col": c,
                "count": det_count,
                "skipped": False,
                "black_ratio": round(float(black_ratio), 1),
            })

    return annotated, tile_results, (rows, cols), box_details


# =========================
# Mode selection
# =========================
st.subheader("1️⃣ Analysis mode")

mode = st.radio(
    "Select analysis target",
    ["Single area", "All areas (batch)"],
    horizontal=True,
)


# =========================
# Single area mode
# =========================
if mode == "Single area":
    file_names = [Path(f).name for f in image_files]

    selected_idx = st.selectbox(
        "Select area",
        range(len(file_names)),
        format_func=lambda i: file_names[i],
    )

    selected_file = image_files[selected_idx]

    if st.button("🚀 Run detection", use_container_width=True):
        with st.spinner("Splitting tiles → Running YOLO inference..."):
            annotated, tile_results, grid_shape, box_details = split_and_detect(
                selected_file,
                model,
                TILE_W,
                TILE_H,
                conf_thres,
                imgsz,
                black_threshold,
                box_thickness,
            )

        st.session_state["single_annotated"] = annotated
        st.session_state["single_tile_results"] = tile_results
        st.session_state["single_grid_shape"] = grid_shape
        st.session_state["single_box_details"] = box_details
        st.session_state["single_selected_file"] = selected_file

    if (
        "single_annotated" in st.session_state
        and st.session_state["single_selected_file"] == selected_file
    ):
        annotated = st.session_state["single_annotated"]
        tile_results = st.session_state["single_tile_results"]
        grid_shape = st.session_state["single_grid_shape"]
        box_details = st.session_state["single_box_details"]

        df_tiles = pd.DataFrame(tile_results)
        df_boxes = pd.DataFrame(box_details) if box_details else pd.DataFrame()

        total_count = int(df_tiles["count"].sum())
        skipped = int(df_tiles["skipped"].sum())
        processed = int(len(df_tiles) - skipped)

        st.subheader("2️⃣ Detection results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🥬 Total detected", total_count)
        m2.metric("Total tiles", len(df_tiles))
        m3.metric("Processed", processed)
        m4.metric("Skipped", skipped)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original image**")
            orig = Image.open(selected_file).convert("RGB")
            st.image(orig, use_container_width=True)

        with col2:
            st.markdown("**Detection result**")
            st.image(annotated, use_container_width=True)

        if not df_boxes.empty:
            st.subheader("📏 Bounding box size analysis")

            st.markdown(
                "Smaller boxes may indicate immature cabbages that are "
                "**not yet ready for harvest**. "
                "Use the statistics below to identify an appropriate size threshold."
            )

            st.markdown("**Box area statistics (px²)**")

            stats = df_boxes["box_area"].describe()

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Count", int(stats["count"]))
            s2.metric("Mean", f"{stats['mean']:.0f}")
            s3.metric("Median", f"{df_boxes['box_area'].median():.0f}")
            s4.metric("Min", f"{stats['min']:.0f}")
            s5.metric("Max", f"{stats['max']:.0f}")

            fig_h, ax_h1 = plt.subplots(figsize=(7, 4))
            ax_h1.hist(
                df_boxes["box_area"],
                bins=30,
                alpha=0.8,
                edgecolor="white",
            )
            ax_h1.axvline(
                df_boxes["box_area"].median(),
                linestyle="--",
                label=f"Median: {df_boxes['box_area'].median():.0f}",
            )
            ax_h1.set_xlabel("Box area (px²)")
            ax_h1.set_ylabel("Frequency")
            ax_h1.set_title("Box area distribution")
            ax_h1.legend()
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

            fig_s, ax_s = plt.subplots(figsize=(7, 4))
            ax_s.hist(
                df_boxes["box_w"],
                bins=25,
                alpha=0.7,
                edgecolor="white",
                label="Width",
            )
            ax_s.hist(
                df_boxes["box_h"],
                bins=25,
                alpha=0.7,
                edgecolor="white",
                label="Height",
            )
            ax_s.set_xlabel("Size (px)")
            ax_s.set_ylabel("Frequency")
            ax_s.set_title("Box width & height distribution")
            ax_s.legend()
            fig_s.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

            st.markdown("**Filter by minimum box area**")

            min_area = st.slider(
                "Minimum box area (px²) — boxes below this are considered immature",
                min_value=0,
                max_value=int(df_boxes["box_area"].max()),
                value=int(df_boxes["box_area"].quantile(0.25)),
                step=100,
            )

            harvest_ready = df_boxes[df_boxes["box_area"] >= min_area]
            immature = df_boxes[df_boxes["box_area"] < min_area]

            hr1, hr2, hr3 = st.columns(3)
            hr1.metric("🥬 Harvest-ready", len(harvest_ready))
            hr2.metric("🌱 Immature / small", len(immature))
            hr3.metric(
                "Harvest ratio",
                f"{len(harvest_ready) / len(df_boxes) * 100:.1f}%",
            )

            with st.expander("View all box details"):
                df_boxes_display = df_boxes.copy()
                df_boxes_display["harvest_ready"] = (
                    df_boxes_display["box_area"] >= min_area
                )
                st.dataframe(df_boxes_display, use_container_width=True)

            csv_boxes = df_boxes.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Download box details CSV",
                data=csv_boxes,
                file_name="cabbage_box_details.csv",
                mime="text/csv",
            )

        st.subheader("3️⃣ Tile heatmap")

        rows, cols = grid_shape
        grid = np.zeros((rows, cols), dtype=int)

        for _, row_data in df_tiles.iterrows():
            grid[int(row_data["row"]), int(row_data["col"])] = int(row_data["count"])

        fig, ax = plt.subplots(
            figsize=(max(cols * 1.5, 6), max(rows * 1.2, 4))
        )

        sns.heatmap(
            grid,
            annot=True,
            fmt="d",
            cmap="YlGn",
            xticklabels=[f"C{i + 1}" for i in range(cols)],
            yticklabels=[f"R{i + 1}" for i in range(rows)],
            ax=ax,
            linewidths=0.5,
        )

        ax.set_title(f"{Path(selected_file).name} — tile detection count")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("4️⃣ Tile data")
        st.dataframe(df_tiles, use_container_width=True)

        csv_tiles = df_tiles.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Download tile results CSV",
            data=csv_tiles,
            file_name="cabbage_tile_results.csv",
            mime="text/csv",
        )


# =========================
# All areas mode
# =========================
else:
    st.markdown(f"Processing **{len(image_files)} areas** sequentially.")

    st.warning(
        "Batch processing may take some time on Streamlit Cloud. "
        "If the app becomes slow, use Single area mode first."
    )

    if st.button("🚀 Run detection on all areas", use_container_width=True):
        all_results = []
        progress = st.progress(0, text="Processing...")

        for idx, fpath in enumerate(image_files):
            fname = Path(fpath).name

            progress.progress(
                idx / len(image_files),
                text=f"Processing: {fname} ({idx + 1}/{len(image_files)})",
            )

            annotated, tile_results, grid_shape, box_details = split_and_detect(
                fpath,
                model,
                TILE_W,
                TILE_H,
                conf_thres,
                imgsz,
                black_threshold,
                box_thickness,
            )

            df_tiles = pd.DataFrame(tile_results)
            total = int(df_tiles["count"].sum())

            nums = "".join(c for c in Path(fpath).stem if c.isdigit())
            fid = int(nums) if nums else idx + 1

            all_results.append({
                "fid": fid,
                "filename": fname,
                "total_count": total,
                "tiles_processed": int((~df_tiles["skipped"]).sum()),
                "tiles_skipped": int(df_tiles["skipped"].sum()),
            })

        progress.progress(1.0, text="Done!")

        df_all = pd.DataFrame(all_results).sort_values("fid")

        st.subheader("2️⃣ Detection summary — All areas")

        m1, m2, m3 = st.columns(3)
        m1.metric("🥬 Total detected", int(df_all["total_count"].sum()))
        m2.metric("Areas", len(df_all))
        m3.metric("Average per area", f"{df_all['total_count'].mean():.0f}")

        st.subheader("3️⃣ Detection count per area")

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(df_all["fid"], df_all["total_count"], alpha=0.8)

        ax.set_xlabel("Area FID")
        ax.set_ylabel("Cabbage count")
        ax.set_title("Cabbage detection count per area")
        ax.set_xticks(df_all["fid"])
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, df_all["total_count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("4️⃣ Field map heatmap")

        st.markdown("Arrange FIDs to match your field layout.")

        default_layout = (
            "10,2\n"
            "5,12,8,3\n"
            "13,6\n"
            "16,18\n"
            "19,11,7,4\n"
            "17,1\n"
            "9\n"
            "14,15\n"
            "20"
        )

        layout_text = st.text_area(
            "Grid layout (comma-separated, newline for each row)",
            value=default_layout,
            height=200,
            help="Match the field layout in QGIS. Use 0 for empty cells.",
        )

        try:
            layout_rows = layout_text.strip().split("\n")
            max_cols = max(len(row.split(",")) for row in layout_rows)

            grid_data = np.zeros((len(layout_rows), max_cols), dtype=int)
            grid_labels = np.full((len(layout_rows), max_cols), "", dtype=object)

            fid_to_count = dict(zip(df_all["fid"], df_all["total_count"]))

            for r, row_str in enumerate(layout_rows):
                cells = row_str.strip().split(",")

                for c, cell in enumerate(cells):
                    cell = cell.strip()

                    if cell and cell != "0":
                        fid = int(cell)
                        count = fid_to_count.get(fid, 0)

                        grid_data[r, c] = count
                        grid_labels[r, c] = f"FID{fid}\n{count}"

            mask = grid_data == 0

            for r in range(len(layout_rows)):
                for c in range(max_cols):
                    if grid_labels[r, c] == "":
                        mask[r, c] = True

            fig2, ax2 = plt.subplots(
                figsize=(max_cols * 2, len(layout_rows) * 1.5)
            )

            sns.heatmap(
                grid_data,
                annot=grid_labels,
                fmt="",
                cmap="YlGn",
                mask=mask,
                ax=ax2,
                linewidths=1,
                linecolor="white",
                cbar_kws={"label": "Detection count"},
            )

            ax2.set_title("Cabbage detection — Field map")
            ax2.set_xticks([])
            ax2.set_yticks([])

            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        except Exception as e:
            st.warning(f"Failed to parse grid layout: {e}")

        st.subheader("5️⃣ Detailed data")
        st.dataframe(df_all, use_container_width=True)

        csv = df_all.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "📥 Download results CSV",
            data=csv,
            file_name="cabbage_detection_all_areas.csv",
            mime="text/csv",
        )