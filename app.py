import streamlit as st
import pandas as pd
from PIL import Image, ImageFile
from datetime import datetime
import json
import os
from pathlib import Path
import time
import base64

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "baseera_logo.png"
FOOD_IMG_DIR = ASSETS_DIR / "foods"

DATA_DIR = Path("food_data")
IMAGES_DIR = DATA_DIR / "images"
DATASET_FILE = DATA_DIR / "dataset.json"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
page_icon = "B"
if LOGO_PATH.exists():
    try:
        page_icon = Image.open(LOGO_PATH)
    except Exception:
        page_icon = "B"

st.set_page_config(
    page_title="Baseera",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Query Params helpers
# -----------------------------------------------------------------------------
def get_qp() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def set_qp(**kwargs):
    try:
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

# -----------------------------------------------------------------------------
# Base64 helper
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def img_to_data_uri(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists():
        return None

    ext = path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}" if ext in ["png", "jpeg", "webp"] else "image/png"

    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# -----------------------------------------------------------------------------
# YOLO import
# -----------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

@st.cache_resource
def load_model():
    model_path = "best_model.pt"
    if not YOLO_AVAILABLE:
        return None
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

MODEL = load_model()

# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def load_dataset():
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_dataset(data):
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_image(uploaded_file, pred_id: str):
    img_path = IMAGES_DIR / f"{pred_id}.jpg"
    Image.open(uploaded_file).convert("RGB").save(img_path)
    return str(img_path)

def classify_image(image: Image.Image):
    if MODEL is None:
        return "Error", 0.0
    try:
        temp_path = "temp_image.jpg"
        image.convert("RGB").save(temp_path)
        results = MODEL.predict(source=temp_path, verbose=False)
        r = results[0]
        probs = r.probs.data.cpu().numpy()
        names = r.names
        top_idx = int(probs.argmax())
        result = str(names[top_idx])
        confidence = float(probs[top_idx] * 100.0)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return result, confidence
    except Exception:
        return "Error", 0.0

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
DISHES = {
    "pizza": {"name": "Pizza", "img": str(FOOD_IMG_DIR / "pizza.jpg")},
    "salad": {"name": "Salad", "img": str(FOOD_IMG_DIR / "salad.jpg")},
    "fries": {"name": "Fries", "img": str(FOOD_IMG_DIR / "fries.jpg")},
    "pasta": {"name": "Pasta", "img": str(FOOD_IMG_DIR / "pasta.jpg")},
}


DASHBOARD_DIR = ASSETS_DIR / "Dashboard"
DASHBOARD_IMAGES = [
    {"title": "Plate Condition Breakdown by Dish ID", "file": "Plate Condition Breakdown by Dish ID.png"},
    {"title": "Overall Plate Condition Distribution", "file": "Overall Plate Condition Distribution.png"},
    {"title": "Uneaten Plate Rate by Meal Category", "file": "Uneaten Plate Rate by Meal Category.png"},
    {"title": "Clean vs. Uneaten Rate by Dish ID", "file": "Clean vs. Uneaten Rate by Dish ID.png"},
]

# -----------------------------------------------------------------------------
# Session
# -----------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "selected_dish" not in st.session_state:
    st.session_state.selected_dish = None
if "predictions" not in st.session_state:
    st.session_state.predictions = load_dataset()

# -----------------------------------------------------------------------------
# Query param click handler
# -----------------------------------------------------------------------------
qp = get_qp()
dish_from_qp = qp.get("dish", None)
if isinstance(dish_from_qp, list):
    dish_from_qp = dish_from_qp[0] if dish_from_qp else None

if dish_from_qp and dish_from_qp in DISHES:
    st.session_state.selected_dish = dish_from_qp
    st.session_state.step = 2
    set_qp()

# -----------------------------------------------------------------------------
# CSS (Balanced analysis page + keep theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: "Cairo", sans-serif; }

:root{
  --bg: #f6f7fb;
  --card: #ffffff;
  --text: #163a2a;
  --muted: #6b7b74;
  --green: #2e7d32;
  --border: rgba(22,58,42,0.12);
  --shadow: 0 10px 22px rgba(0,0,0,0.06);
  --shadow2: 0 6px 16px rgba(0,0,0,0.05);
}

.stApp { background: var(--bg); }
header[data-testid="stHeader"] {visibility: hidden;}
footer {visibility: hidden;}



/* ✅ make content centered & balanced */
.block-container{
  padding-top: 18px !important;
  max-width: 1180px !important;
}

/* HOME */
.center-stage{ width: 100%; padding-top: 120px; }
.section-title{ font-size: 26px; font-weight: 900; color: var(--text); text-align: center; margin: 0; }
.section-line{ width: 320px; height: 2px; background: #e3e7e4; margin: 12px auto 28px auto; border-radius: 2px; }

.page-header{ text-align: center; margin-top: 10px; margin-bottom: 18px; }
.page-title{ font-size: 26px; font-weight: 900; color: var(--text); margin: 0; }
.page-subtitle{ font-size: 14px; font-weight: 700; color: var(--muted); margin-top: 6px; }
.page-line{ width: 220px; height: 2px; background: #e3e7e4; margin: 12px auto 0 auto; border-radius: 2px; }

/* Food cards */
.card-link{ display:block; text-decoration:none !important; color: inherit !important; outline: none !important; }
.dish-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow2);
  overflow: hidden;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.dish-card:hover{
  transform: translateY(-3px);
  border-color: rgba(46,125,50,0.30);
  box-shadow: 0 14px 28px rgba(0,0,0,0.10), 0 0 0 3px rgba(46,125,50,0.12);
}
.dish-img{ width: 100%; height: 240px; object-fit: cover; display:block; }
.dish-name-below{ font-size: 18px; font-weight: 900; color: var(--text); text-align: center; margin-top: 14px; }

/* Sidebar */
[data-testid="stSidebar"]{ background: #ffffff; border-left: 1px solid var(--border); }
[data-testid="stSidebar"] *{ color: var(--text); }
.sb-brand-centered{ display:flex; flex-direction:column; align-items:center; justify-content:center; gap: 8px; margin-top: 6px; }
.sb-brand-title{ font-size: 20px; font-weight: 900; color: var(--text); text-align: center; letter-spacing: 0.3px; }
.sb-brand-title span{ font-weight: 600; }
.sb-divider{ height: 1px; width: 100%; background: rgba(22,58,42,0.10); margin: 10px 0 12px 0; }
.sb-title{ font-weight: 900; font-size: 22px; margin: 0 0 8px 0; }

[data-testid="stSidebar"] .stRadio label{
  width: 100%;
  background: rgba(46,125,50,0.10);
  border: 2px solid rgba(46,125,50,0.22);
  border-radius: 18px;
  padding: 16px 16px;
  margin: 10px 0;
  cursor: pointer;
  transition: transform .16s ease, box-shadow .16s ease, background .16s ease, border-color .16s ease;
  box-shadow: 0 8px 18px rgba(0,0,0,0.05);
}
[data-testid="stSidebar"] .stRadio label > div:first-child{ display: none !important; }
[data-testid="stSidebar"] .stRadio label *{ font-size: 18px !important; font-weight: 800 !important; }
[data-testid="stSidebar"] .stRadio label:hover{
  transform: translateY(-2px);
  background: rgba(46,125,50,0.14);
  border-color: rgba(46,125,50,0.30);
  box-shadow: 0 12px 24px rgba(0,0,0,0.08);
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"]{
  background: rgba(46,125,50,0.16) !important;
  border-color: rgba(46,125,50,0.40) !important;
  box-shadow: 0 14px 28px rgba(0,0,0,0.10), 0 0 0 3px rgba(46,125,50,0.10);
}

.sb-stat{
  background: rgba(46,125,50,0.10);
  border: 3px solid rgba(46,125,50,0.28);
  border-radius: 22px;
  padding: 18px 18px;
  box-shadow: 0 10px 22px rgba(0,0,0,0.06);
  margin-top: 22px;
}
.sb-stat-title{ font-weight: 900; font-size: 20px; margin-bottom: 12px; }
.sb-stat-num{ font-weight: 900; font-size: 46px; color: var(--green); line-height: 1; }

/* ✅ GLOBAL OUTLINE BUTTONS (Fix black buttons) */
div.stButton > button,
div.stDownloadButton > button,
button[kind="secondary"],
button[kind="primary"]{
  width: 100% !important;
  border-radius: 16px !important;
  padding: 0.85rem 1rem !important;
  background: #ffffff !important;
  color: var(--text) !important;
  border: 2px solid rgba(46,125,50,0.55) !important;
  font-weight: 900 !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.06) !important;
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover{
  transform: translateY(-2px) !important;
  border-color: rgba(46,125,50,0.75) !important;
  box-shadow: 0 14px 28px rgba(0,0,0,0.10), 0 0 0 3px rgba(46,125,50,0.10) !important;
  background: rgba(46,125,50,0.06) !important;
}
div.stButton > button:disabled,
div.stDownloadButton > button:disabled,
button:disabled{
  opacity: 0.45 !important;
  cursor: not-allowed !important;
  transform: none !important;
}

/* ✅ Analysis layout */
.analysis-header{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 18px 22px;
}
.analysis-title{ font-size: 22px; font-weight: 900; color: var(--text); margin: 0; }
.analysis-sub{ font-size: 14px; font-weight: 800; color: var(--muted); margin-top: 6px; }

.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 18px;
  height: 100%;
}
.card-title{ font-weight: 900; color: var(--text); margin: 0 0 10px 0; }

.badge{
  display:inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 14px;
  background: rgba(22,58,42,0.06);
  border: 1px solid rgba(22,58,42,0.12);
  color: var(--text);
}
.badge.clean{ background: rgba(46,125,50,0.12); border-color: rgba(46,125,50,0.22); color: #1b5e20; }
.badge.partial{ background: rgba(230,81,0,0.10); border-color: rgba(230,81,0,0.22); color: #e65100; }
.badge.uneaten{ background: rgba(183,28,28,0.10); border-color: rgba(183,28,28,0.22); color: #b71c1c; }

.conf-row{ margin-top: 14px; font-weight: 800; color: var(--muted); }
.progress-track{
  width: 100%; height: 10px; border-radius: 999px;
  background: rgba(22,58,42,0.10);
  overflow: hidden; margin-top: 8px;
}
.progress-bar{ height: 100%; border-radius: 999px; background: rgba(46,125,50,0.65); width: 0%; }
.ts{ margin-top: 12px; font-size: 12px; color: var(--muted); font-weight: 800; }

/* uploader look */
[data-testid="stFileUploader"] section{
  padding: 14px !important;
  border-radius: 16px !important;
  border: 1px dashed rgba(22,58,42,0.22) !important;
  background: rgba(46,125,50,0.06) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sb-brand-centered'>", unsafe_allow_html=True)
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=220)
    st.markdown("<div class='sb-brand-title'>بَصيرة <span>| Baseera</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-title'>Menu</div>", unsafe_allow_html=True)

    page = st.radio("Menu", ["Home", "Dataset", "About"], label_visibility="collapsed")

    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sb-stat">
            <div class="sb-stat-title">Total Analyses</div>
            <div class="sb-stat-num">{len(st.session_state.predictions)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def page_header(title: str, subtitle: str = ""):
    sub_html = f"<div class='page-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">{title}</div>
      {sub_html}
      <div class="page-line"></div>
    </div>
    """, unsafe_allow_html=True)

def render_food_selection_home():
    st.markdown("<div class='center-stage'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Select Food Type</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-line'></div>", unsafe_allow_html=True)

    cols = st.columns(4, gap="large")
    for idx, (dish_id, dish) in enumerate(DISHES.items()):
        with cols[idx]:
            data_uri = img_to_data_uri(dish["img"])
            if data_uri:
                st.markdown(f"""
                <div>
                  <a class="card-link" href="?dish={dish_id}">
                    <div class="dish-card">
                      <img class="dish-img" src="{data_uri}" alt="{dish_id}" />
                    </div>
                  </a>
                  <div class="dish-name-below">{dish['name']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                  <a class="card-link" href="?dish={dish_id}">
                    <div class="dish-card">
                      <div style="height:240px;background:rgba(0,0,0,0.05);display:flex;align-items:center;justify-content:center;">
                        <div style="font-weight:800;color:var(--muted);">Image not found</div>
                      </div>
                    </div>
                  </a>
                  <div class="dish-name-below">{dish['name']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def render_analysis_view():
    dish_id = st.session_state.selected_dish
    dish = DISHES[dish_id]

    # Header (compact)
    st.markdown(f"""
    <div class="analysis-header">
      <div class="analysis-title">Food Analysis</div>
      <div class="analysis-sub">Selected: {dish['name']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Actions row (balanced)
    a1, a2, a3 = st.columns([1, 1, 1], gap="large")

    with a1:
        if st.button("Back to food selection", use_container_width=True):
            st.session_state.step = 1
            st.session_state.selected_dish = None
            st.rerun()

    with a2:
        if st.session_state.predictions:
            df = pd.DataFrame(st.session_state.predictions)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Export CSV", csv, "dataset.csv", "text/csv", use_container_width=True)
        else:
            st.button("Export CSV", use_container_width=True, disabled=True)

    with a3:
        if st.button("Clear History", use_container_width=True, disabled=not bool(st.session_state.predictions)):
            st.session_state.predictions = []
            save_dataset([])
            st.rerun()

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # Main area (two equal cards)
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Image</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload food image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if uploaded:
            if st.button("Analyze Now", use_container_width=True):
                with st.spinner("Processing..."):
                    time.sleep(0.8)
                    image = Image.open(uploaded)
                    result, confidence = classify_image(image)

                    pred_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    img_path = save_image(uploaded, pred_id)

                    prediction = {
                        "id": pred_id,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "dish": dish["name"],
                        "image": img_path,
                        "result": result,
                        "confidence": round(confidence, 2),
                    }
                    st.session_state.predictions.insert(0, prediction)
                    save_dataset(st.session_state.predictions)

                st.success("Done")
                st.rerun()
        else:
            st.button("Analyze Now", use_container_width=True, disabled=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Result</div>", unsafe_allow_html=True)

        if st.session_state.predictions and st.session_state.predictions[0]["dish"] == dish["name"]:
            latest = st.session_state.predictions[0]
            res = str(latest["result"]).strip()
            res_lower = res.lower()

            if res_lower == "clean":
                badge_class = "clean"
            elif "partial" in res_lower:
                badge_class = "partial"
            elif "uneaten" in res_lower:
                badge_class = "uneaten"
            else:
                badge_class = ""

            conf = float(latest["confidence"])
            conf_pct = max(0.0, min(100.0, conf))

            st.markdown(f"""<div class="badge {badge_class}">{res}</div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="conf-row">
              Confidence: <span style="font-weight:900;color:var(--green);">{conf_pct:.2f}%</span>
              <div class="progress-track">
                <div class="progress-bar" style="width:{conf_pct:.2f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='ts'>{latest['date']} - {latest['time']}</div>", unsafe_allow_html=True)
        else:
            st.info("Upload an image, then click Analyze Now to see the result.")

        st.markdown("</div>", unsafe_allow_html=True)

def render_dataset_page():
    page_header("Dataset", "Dashboard preview (static images)")

    DASHBOARD_ITEMS = [
        ("Plate Condition Breakdown by Dish ID", "Plate Condition Breakdown by Dish ID.png"),
        ("Overall Plate Condition Distribution", "Overall Plate Condition Distribution.png"),
        ("Uneaten Plate Rate by Meal Category", "Uneaten Plate Rate by Meal Category.png"),
        ("Clean vs. Uneaten Rate by Dish ID", "Clean vs. Uneaten Rate by Dish ID.png"),
    ]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    rows = [st.columns(2, gap="large"), st.columns(2, gap="large")]
    idx = 0

    for r in rows:
        for col in r:
            title, filename = DASHBOARD_ITEMS[idx]
            img_path = DASHBOARD_DIR / filename

            with col:
                st.markdown(
                    f"""
                    <div class="simple-card" style="padding:16px;">
                      <div style="
                        font-weight:900;
                        font-size:16px;
                        color:var(--text);
                        margin-bottom:10px;
                        text-align:center;
                      ">
                        {title}
                      </div>
                    """,
                    unsafe_allow_html=True
                )

                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"Missing image: {filename}")

                st.markdown("</div>", unsafe_allow_html=True)

            idx += 1

    # ---------- Records ----------
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='simple-card'><div style='font-weight:900;font-size:18px;'>Records</div></div>",
        unsafe_allow_html=True
    )

    if not st.session_state.predictions:
        st.info("No data yet. Start by analyzing images from the Home page.")
        return


    df = pd.DataFrame(st.session_state.predictions)
    view = df[["date", "time", "dish", "result", "confidence"]]

    # ✅ Table CSS (light theme)
    st.markdown("""
    <style>
    /* records table wrapper */
    .records-wrap { margin-top: 10px; }

    /* the actual table */
    table.records-table{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #ffffff;
    border: 1px solid rgba(22,58,42,0.12);
    border-radius: 14px;
    overflow: hidden;
    font-family: "Cairo", sans-serif;
    }

    /* header */
    table.records-table thead th{
    background: #f2f4f3;
    color: #163a2a;
    font-weight: 900;
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid rgba(22,58,42,0.12);
    }

    /* body cells */
    table.records-table tbody td{
    background: #ffffff;
    color: #163a2a;
    font-weight: 800;
    padding: 12px 14px;
    border-bottom: 1px solid rgba(22,58,42,0.08);
    }

    /* zebra rows */
    table.records-table tbody tr:nth-child(even) td{
    background: #fafbfa;
    }

    /* last row border */
    table.records-table tbody tr:last-child td{
    border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # ✅ Build styled HTML table
    styled = (
        view.style
        .hide(axis="index")
        .set_table_attributes('class="records-table"')
    )

    st.markdown(f"<div class='records-wrap'>{styled.to_html()}</div>", unsafe_allow_html=True)



def render_about_page():
    page_header("About Baseera", "Project overview and purpose.")
    st.markdown("""
    <div class="card">
      <div style="font-weight:900;color:var(--text);font-size:18px;">Overview</div>
      <div style="height:8px;"></div>
      <div style="color:var(--muted);font-weight:800;line-height:2;">
        Baseera is a clean dashboard that helps analyze food plate images using a YOLO model.
        The interface is designed to be modern and portfolio-ready.
      </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
if page == "Home":
    if st.session_state.step == 1:
        render_food_selection_home()
    else:
        render_analysis_view()

elif page == "Dataset":
    render_dataset_page()

elif page == "About":
    render_about_page()
