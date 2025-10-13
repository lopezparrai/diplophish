# streamlit_phishing_app.py
# ---------------------------------------------------------------
# Detector de phishing — versión pulida y minimalista (final)
# Requisitos: streamlit, pandas, numpy, plotly, joblib (o pickle)
# ---------------------------------------------------------------

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re
from urllib.parse import urlparse

# ===================== Configuración de página =====================
st.set_page_config(page_title="¿Es phishing o no?", page_icon="🕵️", layout="centered")

# ===================== Estilos globales ============================
st.markdown("""
<style>
/* Ocultar sidebar y progress residuales */
[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stProgress"] { display:none !important; }

/* --- Contenedor --- */
html, body, [data-testid="stAppViewContainer"] { height: 100%; background-color: white !important; }
.block-container {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding-top: .5rem !important;
    padding-bottom: 1rem !important;
    max-width: 700px;
    margin: auto;
    min-height: 90vh;
}

/* --- Eliminar el borde gris del formulario --- */
form[data-testid="stForm"] {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* --- Botón Analizar --- */
div.stButton > button:first-child,
form button,
form button[kind="primary"] {
    background-color: #2563eb !important;  /* Azul base */
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    height: 2.9em !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.20) !important;
    transition: all .18s ease-in-out !important;
}
div.stButton > button:first-child:hover,
form button:hover,
form button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.30) !important;
}
div.stButton > button:first-child:active,
form button:active,
form button[kind="primary"]:active {
    background-color: #1e3a8a !important;
    transform: scale(0.99) !important;
}

/* --- Banner de resultado --- */
.result-banner {
  border-radius: 16px;
  padding: 18px 24px;
  text-align: center;
  font-size: 2rem;
  font-weight: 900;
  letter-spacing: .5px;
  box-shadow: 0 8px 24px rgba(30, 42, 70, 0.12), 0 2px 6px rgba(30, 42, 70, 0.06);
  margin-top: 12px;
  animation: fadeIn .6s ease-in-out;
}
.result-ok    { background: #e6f7ed; color: #0f5132; border: 1px solid #badbcc; }
.result-alert { background: #fdecea; color: #842029; border: 1px solid #f5c2c7; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* --- Footer sutil --- */
.footer {
  font-size: 0.78rem;
  color: #9aa3af;
  text-align: center;
  margin-top: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Forzar color azul del botón Analizar incluso si el tema es rojo */
div.stButton > button:first-child,
form button,
form button[kind="primary"] {
    background-color: #2563eb !important;   /* azul */
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
    transition: all .2s ease-in-out !important;
}
div.stButton > button:first-child:hover,
form button:hover,
form button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;   /* azul más oscuro */
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(29,78,216,0.35);
}
div.stButton > button:first-child:active,
form button:active,
form button[kind="primary"]:active {
    background-color: #1e3a8a !important;   /* azul profundo */
    transform: scale(0.99);
}

/* Desactivar colores de tema Streamlit */
button[data-testid="baseButton-primary"] {
    background-color: #2563eb !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ===================== Encabezado =====================
st.markdown("<h1 style='text-align:center;'>¿Es phishing o no?</h1>", unsafe_allow_html=True)

# ===================== Paths =====================
MODEL_PATH = Path("models/xgb_phishing.pkl")
FEATURE_ORDER_PATH = Path("feature_order.json")
SCALER_PATH = Path("models/standard_scaler.pkl")

# ===================== Import de funciones de features =====================
try:
    from features import procesar_dominio_basico, enriquecer_dominio_scraping
except Exception:
    st.error("No se pudieron cargar las funciones de extracción. Revisá `features.py`.")
    st.stop()

# ===================== Columnas =====================
TRAIN_FEATURES_FALLBACK = [
    'url_length','num_dashes','num_digits','num_special_chars','path_segments','num_dots',
    'hostname_length','path_length','query_length','num_underscores','num_dashes_in_hostname',
    'title_length','http_status_code','has_https','has_ssl_cert','iframe_present','insecure_forms',
    'submit_info_to_email','abnormal_form_action','double_slash_in_path','is_registered_in_ar','responds'
]
NUMERIC_FEATURES = [
    'url_length','num_dashes','num_digits','num_special_chars','path_segments','num_dots',
    'num_underscores','num_dashes_in_hostname','hostname_length','path_length','query_length',
    'site_age_years','time_to_expire_years','response_time','http_status_code','title_length',
    'sensitive_words_count'
]

# ===================== Carga de modelo/scaler =====================
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_scaler(path: Path):
    try:
        import joblib
        if path.exists():
            return joblib.load(path)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def load_feature_order(path: Path, _model) -> List[str]:
    if path.exists():
        try:
            order = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(order, list): return [str(c) for c in order]
        except Exception:
            pass
    if hasattr(_model, "feature_names_in_") and getattr(_model, "feature_names_in_", None) is not None:
        return [str(c) for c in _model.feature_names_in_]
    return list(TRAIN_FEATURES_FALLBACK)

# ===================== Normalización de URL =====================
DOMAIN_REGEX = re.compile(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
def normalize_to_domain(text: str) -> str:
    text = text.strip()
    if not text: return ""
    if not text.startswith(("http://", "https://")):
        if DOMAIN_REGEX.match(text): return text
        text = "http://" + text
    parsed = urlparse(text)
    host = (parsed.netloc or parsed.path).split(":")[0]
    return host[4:] if host.startswith("www.") else host

# ===================== Utilidades de features =====================
def ensure_feature_vector(feat_map: Dict[str, float], feature_order: List[str]) -> pd.DataFrame:
    def _cast(v):
        if v is None: return 0.0
        if isinstance(v, bool): return 1.0 if v else 0.0
        try: return float(v)
        except Exception: return 0.0
    return pd.DataFrame([{k:_cast(feat_map.get(k,0.0)) for k in feature_order}], columns=feature_order)

# ===================== Tacómetro (degradado verde→rojo) =====================
def make_gradient_steps(n: int = 220, vmin: float = 0.0, vmax: float = 100.0):
    stops = [(0.00,(33,197,93)), (0.50,(250,204,21)), (1.00,(239,68,68))]  # verde→amarillo→rojo
    def lerp(a,b,t): return a+(b-a)*t
    def interp_color(p: float):
        for (p0,c0),(p1,c1) in zip(stops[:-1],stops[1:]):
            if p<=p1:
                t=(p-p0)/(p1-p0)
                r=int(lerp(c0[0],c1[0],t)); g=int(lerp(c0[1],c1[1],t)); b=int(lerp(c0[2],c1[2],t))
                return f"rgb({r},{g},{b})"
        r,g,b=stops[-1][1]; return f"rgb({r},{g},{b})"
    span=vmax-vmin; steps=[]
    for i in range(n):
        a=vmin+(i*span)/n; b=vmin+((i+1)*span)/n
        steps.append({"range":[a,b],"color":interp_color((i+0.5)/n)})
    return steps

def render_tacometro(prob: float):
    pct = round(prob * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix":"%", "font":{"size": 46, "color":"#101418", "family":"Arial Black"}},
            gauge={
                "shape":"angular",
                "axis":{"range":[0,100], "tickwidth":0, "ticks":"", "tickvals":[0,20,40,60,80,100],
                        "ticktext":["","","","","",""]},  # sin etiquetas visibles
                "bar":{"color":"rgba(0,0,0,0)","thickness":0},  # sin barra-rail
                "threshold":{"line":{"color":"#111","width":6}, "thickness":0.9, "value":pct},  # “aguja”
                "borderwidth":0, "bgcolor":"rgba(0,0,0,0)",
                "steps": make_gradient_steps(n=220, vmin=0, vmax=100),
            },
            domain={"x":[0,1], "y":[0,1]},
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260)
    fig.layout.margin = go.layout.Margin(l=0, r=0, t=0, b=0)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ===================== Cargar artefactos =====================
try:
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    feature_order = load_feature_order(FEATURE_ORDER_PATH, model)
except Exception:
    st.error("No pudimos cargar el modelo.")
    st.stop()
if scaler is None:
    st.error("Falta el archivo del scaler.")
    st.stop()

# ===================== Predicción =====================
def predict_and_show(dominio: str):
    with st.spinner("Analizando…"):
        base_feats = procesar_dominio_basico(dominio)
        dyn_feats  = enriquecer_dominio_scraping(dominio)
        feats = {**(base_feats or {}), **(dyn_feats or {})}

        X = ensure_feature_vector(feats, feature_order)

        # columnas para scaler
        if hasattr(scaler, "feature_names_in_") and getattr(scaler,"feature_names_in_",None) is not None:
            scaler_cols = [str(c) for c in scaler.feature_names_in_]
        else:
            scaler_cols = [c for c in feature_order if c in NUMERIC_FEATURES]

        scaler_input = pd.DataFrame(columns=scaler_cols)
        for c in scaler_cols:
            scaler_input[c] = X[c].values if c in X.columns else 0.0

        scaled_array = scaler.transform(scaler_input[scaler_cols])
        overlap_cols = [c for c in scaler_cols if c in X.columns]
        if overlap_cols:
            idx_map = [scaler_cols.index(c) for c in overlap_cols]
            X.loc[:, overlap_cols] = scaled_array[:, idx_map]

        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        y_pred = model.predict(X)
        label = int(y_pred[0]) if hasattr(y_pred, "__iter__") else int(y_pred)
        p_phishing = float(proba[0,1]) if (proba is not None and np.ndim(proba)==2 and proba.shape[1]>=2) else (1.0 if label==1 else 0.0)

    # --- Presentación: tacómetro + resultado grande ---
    render_tacometro(p_phishing)
    if label == 1:
        st.markdown('<div class="result-banner result-alert">PHISHING</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-banner result-ok">NO PHISHING</div>', unsafe_allow_html=True)

# ===================== Interfaz principal (con st.form para soportar ENTER) =====================
with st.form("analyzer_form", clear_on_submit=False):
    url_input = st.text_input(
        "Pegá la URL a analizar:",
        placeholder="https://www.ejemplo.com",
        key="url_input"
    )

    def es_dominio_simple(texto: str) -> bool:
        t = texto.strip().lower()
        if "." not in t:
            return False
        t = t.replace("http://", "").replace("https://", "").split("/")[0]
        return t.replace(".", "").replace("-", "").isalnum() and len(t.split(".")[-1]) >= 2

    valido = es_dominio_simple(url_input)
    analizar = st.form_submit_button("🔍 Analizar", type="primary", use_container_width=True)

# ===================== Evento de análisis =====================
if analizar:
    if not url_input.strip():
        st.warning("Ingresá una URL.")
    elif not valido:
        st.warning("Ingresá un dominio/URL válido (ej.: ejemplo.com o https://ejemplo.com).")
    else:
        dominio = normalize_to_domain(url_input)
        if not dominio:
            st.warning("No se pudo interpretar la entrada como dominio/URL.")
        else:
            predict_and_show(dominio)

# ===================== Footer =====================
st.markdown("<div class='footer'>DiploDatos 2025 — Esta herramienta realiza una estimación automática y no garantiza la legitimidad del sitio.</div>", unsafe_allow_html=True)
