import os
import time
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
from rapidfuzz import fuzz
from difflib import SequenceMatcher
import joblib

# Optional map deps (wrapped in try/except later)
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    import folium
    from streamlit_folium import st_folium
    GEO_OK = True
except ImportError:
    GEO_OK = False

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Places Conflation Demo",
    page_icon="🗺️",
    layout="wide"
)

# ==================== GLOBAL STYLE (incl. HIDE OSM ATTR) ====================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #aaa;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .match-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .conflict-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    /* Hide OpenStreetMap attribution bar */
    .leaflet-control-attribution {
        display: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== LOAD MODELS (CACHED) ====================

@st.cache_resource
def load_models():
    """Load embedding models + classifier once per session."""
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    bge_base = SentenceTransformer("BAAI/bge-base-en-v1.5")
    e5_small = SentenceTransformer("intfloat/e5-small-v2")

    clf = None
    threshold = 0.45

    try:
        clf = joblib.load("models/matcher_gb_enhanced.pkl")
        try:
            with open("models/matcher_threshold_enhanced.txt") as f:
                threshold = float(f.read().strip())
        except Exception:
            threshold = 0.4483
    except Exception:
        # Fallback: no classifier file found
        st.warning(
            "⚠️ Could not load `matcher_gb_enhanced.pkl`. "
            "Falling back to a simple similarity-based decision rule."
        )
        clf = None
        threshold = 0.75

    return minilm, bge_base, e5_small, clf, threshold


with st.spinner("Loading models… (first load may be slow)"):
    MINILM, BGE_BASE, E5_SMALL, CLF, THRESHOLD = load_models()

# ==================== HELPER FUNCTIONS ====================

def safe_str(x):
    return "" if x is None or pd.isna(x) else str(x)


def clean_phone(x):
    s = safe_str(x)
    return "".join(ch for ch in s if ch.isdigit())


def get_domain(url):
    s = safe_str(url).strip()
    if not s:
        return ""
    try:
        parsed = urlparse(s)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def embed_single(text, model):
    emb = model.encode(
        [safe_str(text).lower()],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb[0]


def build_features(name_a, addr_a, phone_a, web_a,
                   name_b, addr_b, phone_b, web_b):
    """Rebuild the 30-feature vector used in training, plus pretty metrics."""

    # --- base text variants ---
    text_name_a = safe_str(name_a).lower()
    text_name_b = safe_str(name_b).lower()
    text_name_addr_a = (safe_str(name_a) + ". " + safe_str(addr_a)).lower()
    text_name_addr_b = (safe_str(name_b) + ". " + safe_str(addr_b)).lower()

    # === EMBEDDING FEATURES ===
    # MiniLM
    emb_minilm_name_a = embed_single(text_name_a, MINILM)
    emb_minilm_name_b = embed_single(text_name_b, MINILM)
    sim_minilm_name = float((emb_minilm_name_a * emb_minilm_name_b).sum())

    emb_minilm_na_a = embed_single(text_name_addr_a, MINILM)
    emb_minilm_na_b = embed_single(text_name_addr_b, MINILM)
    sim_minilm_nameaddr = float((emb_minilm_na_a * emb_minilm_na_b).sum())

    # BGE-base
    emb_bge_name_a = embed_single(text_name_a, BGE_BASE)
    emb_bge_name_b = embed_single(text_name_b, BGE_BASE)
    sim_bge_name = float((emb_bge_name_a * emb_bge_name_b).sum())

    emb_bge_na_a = embed_single(text_name_addr_a, BGE_BASE)
    emb_bge_na_b = embed_single(text_name_addr_b, BGE_BASE)
    sim_bge_nameaddr = float((emb_bge_na_a * emb_bge_na_b).sum())

    # E5-small
    emb_e5_name_a = embed_single(text_name_a, E5_SMALL)
    emb_e5_name_b = embed_single(text_name_b, E5_SMALL)
    sim_e5_name = float((emb_e5_name_a * emb_e5_name_b).sum())

    sim_name_avg = (sim_minilm_name + sim_bge_name + sim_e5_name) / 3.0
    sim_name_max = max(sim_minilm_name, sim_bge_name, sim_e5_name)
    sim_nameaddr_avg = (sim_minilm_nameaddr + sim_bge_nameaddr) / 2.0

    # === STRING SIMILARITY ===
    exact_name_match = int(text_name_a.strip() == text_name_b.strip())

    name_fuzz_ratio = fuzz.ratio(text_name_a, text_name_b) / 100.0
    name_fuzz_partial = fuzz.partial_ratio(text_name_a, text_name_b) / 100.0
    name_fuzz_token_sort = fuzz.token_sort_ratio(text_name_a, text_name_b) / 100.0
    name_fuzz_token_set = fuzz.token_set_ratio(text_name_a, text_name_b) / 100.0

    addr_a_lower = safe_str(addr_a).lower()
    addr_b_lower = safe_str(addr_b).lower()
    addr_fuzz_ratio = fuzz.ratio(addr_a_lower, addr_b_lower) / 100.0
    name_levenshtein = SequenceMatcher(None, text_name_a, text_name_b).ratio()

    # === CONTACT FEATURES ===
    phone_a_clean = clean_phone(phone_a)
    phone_b_clean = clean_phone(phone_b)

    same_phone = int(
        phone_a_clean != "" and phone_b_clean != "" and phone_a_clean == phone_b_clean
    )

    domain_a = get_domain(web_a)
    domain_b = get_domain(web_b)
    same_website_domain = int(
        domain_a != "" and domain_b != "" and domain_a == domain_b
    )

    both_contacts_match = same_phone * same_website_domain
    any_contact_match = int(same_phone == 1 or same_website_domain == 1)

    # === INTERACTION FEATURES ===
    name_nameaddr_product = sim_bge_name * sim_bge_nameaddr
    ensemble_name_product = sim_name_avg * sim_nameaddr_avg
    fuzz_bge_product = name_fuzz_token_sort * sim_bge_name

    high_name_sim = int(sim_bge_name > 0.85)
    high_nameaddr_sim = int(sim_bge_nameaddr > 0.85)
    phone_and_high_sim = same_phone * high_name_sim
    website_and_high_sim = same_website_domain * high_name_sim

    # === CONFIDENCE FEATURES (we don’t have per-record conf here; set 0) ===
    avg_confidence = 0.0
    min_confidence = 0.0
    confidence_diff = 0.0
    both_high_confidence = 0

    features = np.array([
        sim_minilm_name, sim_minilm_nameaddr,
        sim_bge_name, sim_bge_nameaddr,
        sim_e5_name,
        sim_name_avg, sim_name_max, sim_nameaddr_avg,
        exact_name_match,
        name_fuzz_ratio, name_fuzz_partial,
        name_fuzz_token_sort, name_fuzz_token_set,
        addr_fuzz_ratio, name_levenshtein,
        same_phone, same_website_domain,
        both_contacts_match, any_contact_match,
        name_nameaddr_product, ensemble_name_product,
        fuzz_bge_product,
        high_name_sim, high_nameaddr_sim,
        phone_and_high_sim, website_and_high_sim,
        avg_confidence, min_confidence, confidence_diff, both_high_confidence
    ]).reshape(1, -1)

    display_metrics = {
        "Fuzzy Token Set": f"{name_fuzz_token_set:.3f}",
        "BGE Name+Addr Sim": f"{sim_bge_nameaddr:.3f}",
        "Name×Addr Product": f"{name_nameaddr_product:.3f}",
        "Levenshtein": f"{name_levenshtein:.3f}",
        "Same Phone": "✅" if same_phone else "❌",
        "Same Website": "✅" if same_website_domain else "❌",
    }

    # also return some raw sims so fallback proba can use them if needed
    raw_sims = {
        "sim_name_avg": sim_name_avg,
        "sim_bge_nameaddr": sim_bge_nameaddr,
    }

    return features, display_metrics, raw_sims


def predict_match(features, raw_sims):
    """Predict match using GB classifier if available; else similarity rule."""
    if CLF is not None:
        proba = CLF.predict_proba(features)[0, 1]
        label = int(proba >= THRESHOLD)
        return label, float(proba)

    # Fallback: use average of name & name+addr similarities
    sim = 0.6 * raw_sims["sim_name_avg"] + 0.4 * raw_sims["sim_bge_nameaddr"]
    # Map cosine [0,1] -> [0,1] probability-ish
    proba = max(min(sim, 1.0), 0.0)
    label = int(proba >= THRESHOLD)
    return label, float(proba)

# ==================== HEADER & METRICS ====================

st.markdown("<div class='main-header'>🗺️ Places Conflation Demo</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Gradient-Boosted Ensemble on MiniLM + BGE + E5 · F1 ≈ 0.897 (3-fold CV)</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("F1 Score (CV)", "0.897", "+6.6% vs baseline")
with c2:
    st.metric("Accuracy", "87.4%")
with c3:
    st.metric("AUC", "0.940")
with c4:
    st.metric("Engineered Features", "30")

# ==================== DATASET LOADER ====================

st.markdown("---")
st.markdown("## 🗺️ Load Places from Overture Dataset")

with st.expander("📊 Browse Your Overture Training Dataset", expanded=False):
    st.markdown(
        "Load random place pairs from your 2,731-record Overture dataset "
        "to see how the model behaves on real production data."
    )

    col_load1, col_load2 = st.columns([2, 1])

    with col_load1:
        if st.button("🎲 Load Random Place Pair from Dataset", use_container_width=True):
            try:
                if not os.path.exists("places_cleaned.parquet"):
                    st.error("❌ File `places_cleaned.parquet` not found next to app.py.")
                    st.stop()

                df_dataset = pd.read_parquet("places_cleaned.parquet")
                random_row = df_dataset.sample(n=1).iloc[0]

                # Fill session_state so widgets pick up values
                st.session_state["name_a"] = safe_str(random_row["name"])
                st.session_state["addr_a"] = safe_str(random_row["address"])
                st.session_state["phone_a"] = safe_str(random_row["phone"])
                st.session_state["web_a"] = safe_str(random_row["website"])

                st.session_state["name_b"] = safe_str(random_row["base_name"])
                st.session_state["addr_b"] = safe_str(random_row["base_address"])
                st.session_state["phone_b"] = safe_str(random_row["base_phone"])
                st.session_state["web_b"] = safe_str(random_row["base_website"])

                st.session_state["ground_truth"] = int(random_row["label"])

                # Remember the raw row too if you want to debug later
                st.session_state["last_row"] = random_row.to_dict()

                st.success(f"✅ Loaded dataset with {len(df_dataset)} rows.")
                time.sleep(0.3)
                st.rerun()


            except Exception as e:
                st.error(f"❌ Error loading dataset: {e}")

    with col_load2:
        if "ground_truth" in st.session_state:
            if st.session_state["ground_truth"] == 1:
                st.success("✅ Ground Truth: **MATCH**")
            else:
                st.error("❌ Ground Truth: **NO MATCH**")

# ==================== INPUT SECTION ====================

st.markdown("---")
st.markdown("## ✍️ Edit or Enter Place Details")

# Initialize keys so Streamlit doesn’t complain on first run
for key in ["name_a", "addr_a", "phone_a", "web_a",
            "name_b", "addr_b", "phone_b", "web_b"]:
    st.session_state.setdefault(key, "")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### 📍 Place A")
    name_a = st.text_input("Business Name", key="name_a")
    addr_a = st.text_area("Address", height=80, key="addr_a")
    phone_a = st.text_input("Phone Number", key="phone_a")
    web_a = st.text_input("Website", key="web_a")

with col_right:
    st.markdown("### 📍 Place B")
    name_b = st.text_input("Business Name", key="name_b")
    addr_b = st.text_area("Address", height=80, key="addr_b")
    phone_b = st.text_input("Phone Number", key="phone_b")
    web_b = st.text_input("Website", key="web_b")

st.markdown("###")

# ==================== CLEAR BUTTONS ====================

if (
    "dataset_place_a" in st.session_state
    or "ground_truth" in st.session_state
    or "prediction_results" in st.session_state
):
    col_clear1, col_clear2 = st.columns(2)

    with col_clear1:
        if st.button("🔄 Clear Dataset Pair", use_container_width=True):
            for k in ["ground_truth", "last_row"]:
                st.session_state.pop(k, None)
            st.rerun()


    with col_clear2:
        if st.button("🗑️ Clear Prediction", use_container_width=True):
            st.session_state.pop("prediction_results", None)
            st.rerun()


# ==================== PREDICTION BUTTON ====================

if st.button("🔍 Analyze Match Probability", use_container_width=True):
    if not name_a or not name_b:
        st.error("⚠️ Please enter at least the business names for both places.")
    else:
        with st.spinner("Computing 30 features from 3 embedding models…"):
            t0 = time.time()
            X, metrics_dict, raw_sims = build_features(
                name_a, addr_a, phone_a, web_a,
                name_b, addr_b, phone_b, web_b
            )
            label, proba = predict_match(X, raw_sims)
            elapsed = time.time() - t0

        st.session_state["prediction_results"] = {
            "label": label,
            "proba": proba,
            "elapsed": elapsed,
            "metrics": metrics_dict,
            "features": X,
            "inputs": {
                "name_a": name_a, "addr_a": addr_a,
                "phone_a": phone_a, "web_a": web_a,
                "name_b": name_b, "addr_b": addr_b,
                "phone_b": phone_b, "web_b": web_b,
            },
        }
        st.rerun()


# ==================== DISPLAY RESULTS ====================

if "prediction_results" in st.session_state:
    results = st.session_state["prediction_results"]
    label = results["label"]
    proba = results["proba"]
    elapsed = results["elapsed"]
    metrics_dict = results["metrics"]
    X = results["features"]
    inputs = results["inputs"]

    # Unpack for map section
    name_a = inputs["name_a"]
    addr_a = inputs["addr_a"]
    phone_a = inputs["phone_a"]
    web_a = inputs["web_a"]
    name_b = inputs["name_b"]
    addr_b = inputs["addr_b"]
    phone_b = inputs["phone_b"]
    web_b = inputs["web_b"]

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    expected_label = st.session_state.get("ground_truth", None)

    # --- Expected vs Predicted boxes ---
    if expected_label is not None:
        col_expected, col_vs, col_pred = st.columns([2, 0.5, 2])

        with col_expected:
            st.markdown("### Expected")
            if expected_label == 1:
                text = "✓ MATCH"
            else:
                text = "✗ NO MATCH"
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                            padding:1.5rem;border-radius:15px;text-align:center;'>
                    <h2 style='color:white;margin:0;'>{text}</h2>
                    <p style='color:white;font-size:0.9rem;margin-top:0.5rem;'>Ground Truth</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_vs:
            st.markdown(
                "<div style='text-align:center;padding-top:2rem;font-size:2rem;'>→</div>",
                unsafe_allow_html=True,
            )

        with col_pred:
            st.markdown("### Predicted")
            is_correct = int(label) == int(expected_label)

            if label == 1:
                pred_text = "MATCH"
            else:
                pred_text = "NO MATCH"

            if is_correct:
                gradient = "linear-gradient(135deg,#11998e 0%,#38ef7d 100%)"
                icon = "✅"
            else:
                gradient = "linear-gradient(135deg,#ee0979 0%,#ff6a00 100%)"
                icon = "❌"

            st.markdown(
                f"""
                <div style='background:{gradient};
                            padding:1.5rem;border-radius:15px;text-align:center;'>
                    <h2 style='color:white;margin:0;'>{icon} {pred_text}</h2>
                    <p style='color:white;font-size:0.9rem;margin-top:0.5rem;'>Model Prediction</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if is_correct:
            st.success("### ✅ CORRECT PREDICTION (matches ground truth)")
        else:
            st.error("### ❌ INCORRECT PREDICTION (differs from ground truth)")

    else:
        # No ground truth
        if label == 1:
            gradient = "linear-gradient(135deg,#11998e 0%,#38ef7d 100%)"
            text = "✅ MATCH"
            subtitle = "These places likely represent the same location."
        else:
            gradient = "linear-gradient(135deg,#ee0979 0%,#ff6a00 100%)"
            text = "❌ NO MATCH"
            subtitle = "These places appear to be different locations."

        st.markdown(
            f"""
            <div style='background:{gradient};
                        padding:2rem;border-radius:15px;text-align:center;'>
                <h1 style='color:white;margin:0;'>{text}</h1>
                <p style='color:white;font-size:1.1rem;margin-top:0.5rem;'>{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("###")
    # --- Probability metrics ---
    cprob1, cprob2, cprob3 = st.columns(3)
    with cprob1:
        st.metric("Match Probability", f"{proba:.1%}",
                  delta=f"{abs(proba - THRESHOLD):.1%} from threshold")
        st.progress(float(max(min(proba, 1.0), 0.0)))
    with cprob2:
        st.metric("Decision Threshold", f"{THRESHOLD:.3f}")
        st.caption("Cutoff for MATCH vs NO MATCH")
    with cprob3:
        st.metric("Computation Time", f"{elapsed:.2f}s")
        st.caption("Feature extraction + prediction")

    st.markdown("###")
    # --- Feature thumbnails ---
    st.markdown("### 📊 Top Feature Values")
    cols = st.columns(len(metrics_dict))
    for col, (name, value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(name, value)

    # ==================== MAP SECTION ====================
    st.markdown("---")
    st.markdown("### 🗺️ Geographic Visualization")

    if GEO_OK:
        try:
            geolocator = Nominatim(user_agent="places_conflation_demo", timeout=10)
            with st.spinner("Geocoding addresses…"):
                loc_a = geolocator.geocode(addr_a) if addr_a else None
                loc_b = geolocator.geocode(addr_b) if addr_b else None

            if loc_a and loc_b:
                center_lat = (loc_a.latitude + loc_b.latitude) / 2
                center_lon = (loc_a.longitude + loc_b.longitude) / 2
                dist_km = geodesic(
                    (loc_a.latitude, loc_a.longitude),
                    (loc_b.latitude, loc_b.longitude),
                ).km

                if dist_km < 1:
                    zoom = 15
                elif dist_km < 10:
                    zoom = 12
                elif dist_km < 100:
                    zoom = 9
                else:
                    zoom = 6

                col_map, col_side = st.columns([3, 1])

                with col_map:
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=zoom,
                        tiles="OpenStreetMap",
                    )

                    folium.Marker(
                        [loc_a.latitude, loc_a.longitude],
                        popup=folium.Popup(
                            f"<b>Place A</b><br>{name_a}<br>{addr_a}<br>{phone_a}<br>{web_a}",
                            max_width=300,
                        ),
                        tooltip=f"📍 Place A: {name_a}",
                        icon=folium.Icon(color="blue", icon="circle", prefix="fa"),
                    ).add_to(m)

                    marker_color = "green" if label == 1 else "red"
                    folium.Marker(
                        [loc_b.latitude, loc_b.longitude],
                        popup=folium.Popup(
                            f"<b>Place B</b><br>{name_b}<br>{addr_b}<br>{phone_b}<br>{web_b}",
                            max_width=300,
                        ),
                        tooltip=f"📍 Place B: {name_b}",
                        icon=folium.Icon(color=marker_color, icon="circle", prefix="fa"),
                    ).add_to(m)

                    if label == 0:
                        folium.PolyLine(
                            [[loc_a.latitude, loc_a.longitude],
                             [loc_b.latitude, loc_b.longitude]],
                            color="red",
                            weight=3,
                            opacity=0.7,
                            dash_array="10",
                        ).add_to(m)
                    else:
                        folium.Circle(
                            [loc_a.latitude, loc_a.longitude],
                            radius=100,
                            color="green",
                            fill=True,
                            fill_opacity=0.2,
                        ).add_to(m)

                    st_folium(m, width=700, height=500)

                with col_side:
                    st.markdown("#### 📌 Location Details")

                    st.markdown(
                        """
                        <div class='match-card'>
                            <h4 style='margin:0;color:#007bff;'>🔵 Place A</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Name:** {name_a}")
                    st.markdown(f"**Address:** {addr_a}")
                    if phone_a:
                        st.markdown(f"**Phone:** {phone_a}")
                    if web_a:
                        st.markdown(f"**Website:** {web_a[:40]}…")
                    st.markdown(
                        f"**Coords:** {loc_a.latitude:.4f}, {loc_a.longitude:.4f}"
                    )

                    st.markdown("---")

                    if label == 1:
                        card_class = "match-card"
                        icon = "🟢"
                        status = "MATCHED"
                    else:
                        card_class = "conflict-card"
                        icon = "🔴"
                        status = "DIFFERENT"

                    st.markdown(
                        f"""
                        <div class='{card_class}'>
                            <h4 style='margin:0;'>{icon} Place B – {status}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Name:** {name_b}")
                    st.markdown(f"**Address:** {addr_b}")
                    if phone_b:
                        st.markdown(f"**Phone:** {phone_b}")
                    if web_b:
                        st.markdown(f"**Website:** {web_b[:40]}…")
                    st.markdown(
                        f"**Coords:** {loc_b.latitude:.4f}, {loc_b.longitude:.4f}"
                    )

                    st.markdown("---")
                    st.metric("📏 Distance", f"{dist_km:.2f} km")

                    if dist_km < 0.1:
                        st.success("✓ Very close proximity")
                    elif dist_km < 1:
                        st.info("ℹ️ Same neighborhood")
                    elif dist_km < 50:
                        st.warning("⚠️ Same city/region")
                    else:
                        st.error("❌ Different regions")

            else:
                st.info(
                    "💡 Could not geocode one or both addresses. "
                    "Try including city, state, and country."
                )

        except Exception as e:
            st.warning(f"⚠️ Map unavailable: {e}")
    else:
        st.info("📦 Install `geopy`, `folium`, and `streamlit-folium` to see the map.")

    # ==================== TECHNICAL DETAILS ====================
    with st.expander("🔧 Technical Details"):
        st.markdown(
            f"""
**Pipeline**

1. Text preprocessing (names + addresses, lowercasing)
2. Embedding with 3 models: MiniLM-L6-v2, BGE-base-en-v1.5, E5-small-v2  
3. 30 engineered similarity + fuzzy + contact features  
4. Gradient Boosting classifier (if available) or similarity fallback

**Raw Probability:** {proba:.6f}  
**Threshold:** {THRESHOLD:.6f}  
**Prediction:** {'MATCH' if label == 1 else 'NO MATCH'}  

**Feature Vector Shape:** {X.shape}
"""
        )

# ==================== SIDEBAR ====================

st.sidebar.markdown("## 📈 Enhanced Model Performance")
st.sidebar.markdown(
    """
**3-Fold Cross-Validation**

- **F1 Score:** 0.897  
- **Accuracy:** 87.4%  
- **Precision:** 88.3%  
- **Recall:** 91.2%  
- **AUC:** 0.940  

**Improvement over MiniLM baseline:** +6.65% F1
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏆 Top 5 Features (by importance)")
st.sidebar.markdown(
    """
1. **Fuzzy Token Set** (~46%)  
2. **BGE Name+Addr Similarity** (~15%)  
3. **Name×Addr Product** (~5%)  
4. **Levenshtein Distance** (~4%)  
5. **Confidence Difference** (~3%)  
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏗️ Architecture")
st.sidebar.markdown(
    """
**Embedding Models**

- MiniLM-L6-v2 (384-dim)  
- BGE-base-en-v1.5 (768-dim)  
- E5-small-v2 (384-dim)  

**Classifier**

- Gradient Boosting (300 trees, depth 4, lr=0.05)  

**Total Features:** 30  
**Latency (demo):** ~2–3s per prediction
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Data Source:** Overture Maps – Places dataset (CDLA-Permissive-2.0)  

**Course:** CRWN 102 · Corporate Innovation in Spatial Computing  
**Institution:** UC Santa Cruz
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center;color:#666;font-size:0.9rem;'>"
    "Built with Streamlit & Sentence Transformers"
    "</div>",
    unsafe_allow_html=True,
)
