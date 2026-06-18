import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Content Classifier",
    page_icon="🎬",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #141414; }
    .stApp { background-color: #141414; color: #e5e5e5; }
    h1 { color: #E50914 !important; font-family: 'Georgia', serif; }
    h2, h3 { color: #e5e5e5 !important; }
    .metric-card {
        background: #1c1c1c;
        border: 1px solid #2e2e2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { color: #46d369; font-size: 24px; font-weight: bold; }
    .metric-lbl { color: #888; font-size: 12px; text-transform: uppercase; }
    .result-movie { color: #E50914; font-size: 32px; font-weight: bold; }
    .result-tv { color: #4ea8de; font-size: 32px; font-weight: bold; }
    .stSelectbox label, .stNumberInput label, .stMultiSelect label {
        color: #aaa !important;
    }
    div[data-testid="stMetricValue"] { color: #46d369 !important; }
</style>
""", unsafe_allow_html=True)

# ── TRAIN MODEL (cached) ─────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("netflix_titles.csv")
    df_clean = df.copy()
    df_clean.drop(columns=["director", "cast"], inplace=True)
    df_clean["country"]    = df_clean["country"].fillna("Unknown")
    df_clean["date_added"] = df_clean["date_added"].fillna("Unknown")

    valid_ratings = ["TV-MA","TV-14","TV-PG","R","PG-13","TV-Y7","TV-Y","PG","TV-G","NR","G","TV-Y7-FV","NC-17","UR"]
    df_clean["rating"] = df_clean["rating"].apply(lambda x: x if x in valid_ratings else "NR")
    df_clean.dropna(subset=["duration"], inplace=True)
    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    def parse_duration(row):
        dur = str(row["duration"])
        if "min" in dur:   return int(dur.replace(" min","").strip())
        if "Season" in dur: return int(dur.split(" ")[0].strip())
        return 0

    df_clean["duration_value"] = df_clean.apply(parse_duration, axis=1)

    def extract_year(d):
        try:    return pd.to_datetime(d).year
        except: return 0
    df_clean["year_added"] = df_clean["date_added"].apply(extract_year)

    rating_map = {
        "TV-Y":0,"G":0,"TV-Y7":1,"TV-Y7-FV":1,"TV-G":1,"PG":1,
        "TV-PG":2,"PG-13":2,"TV-14":2,"TV-MA":3,"R":3,"NC-17":3,"NR":2,"UR":2
    }
    df_clean["rating_encoded"] = df_clean["rating"].map(rating_map).fillna(2).astype(int)

    top_genres = [
        "International Movies","Dramas","Comedies","International TV Shows",
        "Documentaries","Action & Adventure","TV Dramas","Independent Movies",
        "Children & Family Movies","Romantic Movies","TV Comedies","Thrillers",
        "Crime TV Shows","Kids' TV","Docuseries"
    ]
    for genre in top_genres:
        col = "genre_" + genre.replace(" ","_").replace("&","and").replace("'","").replace(",","")
        df_clean[col] = df_clean["listed_in"].str.contains(genre, regex=False).astype(int)

    def group_country(c):
        c = str(c)
        if "United States"  in c: return "US"
        if "India"          in c: return "India"
        if "United Kingdom" in c: return "UK"
        if "Japan"          in c: return "Japan"
        if "South Korea"    in c: return "South_Korea"
        if "Unknown"        in c: return "Unknown"
        return "Other"

    df_clean["country_group"]   = df_clean["country"].apply(group_country)
    le = LabelEncoder()
    df_clean["country_encoded"] = le.fit_transform(df_clean["country_group"])
    df_clean["target"]          = (df_clean["type"] == "Movie").astype(int)

    genre_cols   = [c for c in df_clean.columns if c.startswith("genre_")]
    feature_cols = ["duration_value","release_year","year_added","rating_encoded","country_encoded"] + genre_cols

    X = df_clean[feature_cols].copy()
    y = df_clean["target"].copy()

    cols_to_scale = ["duration_value","release_year","year_added"]
    scaler = StandardScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    metrics = {
        "rf": {
            "accuracy":  round(accuracy_score(y_test, rf_pred)*100, 2),
            "precision": round(precision_score(y_test, rf_pred, average="weighted")*100, 2),
            "recall":    round(recall_score(y_test, rf_pred, average="weighted")*100, 2),
            "f1":        round(f1_score(y_test, rf_pred, average="weighted")*100, 2),
        },
        "lr": {
            "accuracy":  round(accuracy_score(y_test, lr_pred)*100, 2),
            "precision": round(precision_score(y_test, lr_pred, average="weighted")*100, 2),
            "recall":    round(recall_score(y_test, lr_pred, average="weighted")*100, 2),
            "f1":        round(f1_score(y_test, lr_pred, average="weighted")*100, 2),
        }
    }

    return rf, lr, scaler, le, feature_cols, genre_cols, metrics

# ── HEADER ───────────────────────────────────────────────
st.markdown("# 🎬 Netflix Content Classifier")
st.markdown("##### Prediksi apakah konten Netflix adalah **Movie** atau **TV Show** menggunakan Machine Learning.")
st.divider()

# ── LOAD MODEL ───────────────────────────────────────────
with st.spinner("Melatih model... (hanya sekali)"):
    try:
        rf_model, lr_model, scaler, le, feature_cols, genre_cols, metrics = load_and_train()
        model_loaded = True
    except FileNotFoundError:
        st.error("❌ File `netflix_titles.csv` tidak ditemukan. Pastikan file ada di folder yang sama dengan `app.py`.")
        model_loaded = False
        st.stop()

# ── METRICS ──────────────────────────────────────────────
st.markdown("### 📊 Performa Model (Test Set 20%)")

tab1, tab2 = st.tabs(["🌲 Random Forest", "📈 Logistic Regression"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi",  f"{metrics['rf']['accuracy']}%")
    c2.metric("Presisi",  f"{metrics['rf']['precision']}%")
    c3.metric("Recall",   f"{metrics['rf']['recall']}%")
    c4.metric("F1-Score", f"{metrics['rf']['f1']}%")

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi",  f"{metrics['lr']['accuracy']}%")
    c2.metric("Presisi",  f"{metrics['lr']['precision']}%")
    c3.metric("Recall",   f"{metrics['lr']['recall']}%")
    c4.metric("F1-Score", f"{metrics['lr']['f1']}%")

st.caption("Model dilatih dengan dataset Netflix Titles (8.807 data, 20 fitur). Random Forest dipilih sebagai model deployment.")
st.divider()

# ── INPUT FORM ───────────────────────────────────────────
st.markdown("### 🔍 Masukkan Detail Konten")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input(
        "Durasi (menit untuk Movie / jumlah season untuk TV Show)",
        min_value=1, max_value=500, value=90,
        help="Contoh: 90 (Movie 90 menit) atau 3 (TV Show 3 season)"
    )
    release_year = st.number_input(
        "Tahun Rilis", min_value=1900, max_value=2025, value=2020
    )
    year_added = st.number_input(
        "Tahun Masuk Netflix", min_value=2008, max_value=2025, value=2021
    )

with col2:
    rating_label = st.selectbox(
        "Rating Konten",
        options=[
            "TV-MA / R / NC-17 (Dewasa)",
            "TV-14 / PG-13 / TV-PG / NR (Remaja)",
            "TV-Y7 / TV-G / PG (Keluarga)",
            "TV-Y / G (Anak-anak)"
        ]
    )
    rating_map_input = {
        "TV-MA / R / NC-17 (Dewasa)": 3,
        "TV-14 / PG-13 / TV-PG / NR (Remaja)": 2,
        "TV-Y7 / TV-G / PG (Keluarga)": 1,
        "TV-Y / G (Anak-anak)": 0
    }
    rating_encoded = rating_map_input[rating_label]

    country_label = st.selectbox(
        "Negara Produksi",
        options=["United States", "India", "United Kingdom", "Japan", "South Korea", "Negara Lain", "Tidak Diketahui"]
    )
    country_map_input = {
        "United States": 5, "India": 0, "United Kingdom": 4,
        "Japan": 1, "South Korea": 3, "Negara Lain": 2, "Tidak Diketahui": 6
    }
    country_encoded = country_map_input[country_label]

# Genre selection
st.markdown("**Genre** (pilih semua yang sesuai)")
genre_options = {
    "genre_International_Movies":        "International Movies",
    "genre_Dramas":                      "Dramas",
    "genre_Comedies":                    "Comedies",
    "genre_International_TV_Shows":      "International TV Shows",
    "genre_Documentaries":               "Documentaries",
    "genre_Action_and_Adventure":        "Action & Adventure",
    "genre_TV_Dramas":                   "TV Dramas",
    "genre_Independent_Movies":          "Independent Movies",
    "genre_Children_and_Family_Movies":  "Children & Family Movies",
    "genre_Romantic_Movies":             "Romantic Movies",
    "genre_TV_Comedies":                 "TV Comedies",
    "genre_Thrillers":                   "Thrillers",
    "genre_Crime_TV_Shows":              "Crime TV Shows",
    "genre_Kids_TV":                     "Kids' TV",
    "genre_Docuseries":                  "Docuseries",
}

selected_genres = st.multiselect(
    "Pilih genre:",
    options=list(genre_options.values()),
    placeholder="Klik untuk memilih genre..."
)
genre_key_selected = [k for k, v in genre_options.items() if v in selected_genres]

st.divider()

# ── PREDICT ──────────────────────────────────────────────
model_choice = st.radio(
    "Gunakan model:",
    options=["🌲 Random Forest (Terbaik)", "📈 Logistic Regression"],
    horizontal=True
)

if st.button("🎬 Prediksi Sekarang", use_container_width=True, type="primary"):

    # Build feature vector
    raw = {
        "duration_value":  duration,
        "release_year":    release_year,
        "year_added":      year_added,
        "rating_encoded":  rating_encoded,
        "country_encoded": country_encoded,
    }
    for k in genre_options.keys():
        raw[k] = 1 if k in genre_key_selected else 0

    features = np.array([[raw[f] for f in feature_cols]])

    # Standardize
    cols_to_scale_idx = [0, 1, 2]
    features[0, cols_to_scale_idx] = scaler.transform(
        features[:, cols_to_scale_idx].reshape(1, -1)
    )[0]

    # Predict
    if "Random Forest" in model_choice:
        model = rf_model
        model_name = "Random Forest"
    else:
        model = lr_model
        model_name = "Logistic Regression"

    pred      = model.predict(features)[0]
    prob      = model.predict_proba(features)[0]
    prob_movie = prob[1]
    prob_tv    = prob[0]

    # Result
    st.markdown("---")
    st.markdown("### 🎯 Hasil Prediksi")

    if pred == 1:
        st.markdown('<p class="result-movie">🎬 MOVIE</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result-tv">📺 TV SHOW</p>', unsafe_allow_html=True)

    st.markdown(f"**Model yang digunakan:** {model_name}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilitas Movie",   f"{prob_movie*100:.1f}%")
    with col_b:
        st.metric("Probabilitas TV Show", f"{prob_tv*100:.1f}%")

    # Progress bars
    st.markdown("**Confidence:**")
    st.markdown(f"🎬 Movie")
    st.progress(float(prob_movie))
    st.markdown(f"📺 TV Show")
    st.progress(float(prob_tv))

    st.success(f"✅ Prediksi selesai! Konten ini kemungkinan besar adalah **{'Movie' if pred == 1 else 'TV Show'}** dengan confidence **{max(prob_movie, prob_tv)*100:.1f}%**")

# ── FOOTER ───────────────────────────────────────────────
st.divider()
st.caption("Netflix Content Classifier · Random Forest · Akurasi 99.94% · AI & Big Data Final Project 2026")
