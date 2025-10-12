# streamlit_phishing_app.py
# ---------------------------------------------------------------
# App para clasificar una URL/Dominio como phishing (1) o no (0)
# Front orientado a usuario final: limpio, guiado y sin jerga técnica.
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

# ----------------------- Config de página -----------------------
st.set_page_config(page_title="Detector de phishing", page_icon="🛡️", layout="centered")
st.title("🛡️ Detector de sitios sospechosos")
st.caption("Pegá una URL y te mostramos el **riesgo estimado** de phishing. No guardamos datos.")

# ==== Estilo “card” para el tacómetro ====
st.markdown("""
<style>
.gauge-card {
  background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 60%, #eef2f7 100%);
  border-radius: 18px;
  box-shadow: 0 8px 24px rgba(30, 42, 70, 0.15), 0 2px 6px rgba(30, 42, 70, 0.08);
  padding: 18px 18px 8px 18px;
  border: 1px solid rgba(0,0,0,0.05);
}
.gauge-title {
  font-size: 0.95rem; color: #5b6b7f; margin: 0 0 6px 2px; letter-spacing: .3px;
}
.gauge-subtitle {
  font-size: 0.85rem; color: #7a8a9f; margin: -6px 0 10px 2px;
}
.small-muted { color:#6b7280; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------- Paths de artefactos --------------------
MODEL_PATH = Path("models/xgb_phishing.pkl")
FEATURE_ORDER_PATH = Path("feature_order.json")
SCALER_PATH = Path("models/standard_scaler.pkl")

# --------------------- Import de tus features -------------------
try:
    from features import procesar_dominio_basico, enriquecer_dominio_scraping
except Exception as e:
    st.error("No se pudieron cargar las funciones de extracción de features. Revisá `features.py`.")
    st.stop()

# ------------------- Conjuntos de columnas base ----------------
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

# --------------------- Carga de modelo/scaler -------------------
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

def get_expected_order(feature_order_json: List[str], _scaler, _model, observed_keys: List[str]) -> List[str]:
    if hasattr(_model, "feature_names_in_") and getattr(_model, "feature_names_in_", None) is not None:
        fins = [str(c) for c in _model.feature_names_in_]
        if fins: return fins
    if feature_order_json: return [str(c) for c in feature_order_json]
    if TRAIN_FEATURES_FALLBACK: return list(TRAIN_FEATURES_FALLBACK)
    if _scaler is not None and hasattr(_scaler, "feature_names_in_") and getattr(_scaler, "feature_names_in_", None) is not None:
        sins = [str(c) for c in _scaler.feature_names_in_]
        if sins: return sins
    return sorted([str(k) for k in observed_keys])

# ------------------ Alineación robusta de vector ----------------
def ensure_feature_vector(feat_map: Dict[str, float], feature_order: List[str]) -> pd.DataFrame:
    def _cast(v):
        if v is None: return 0.0
        if isinstance(v, bool): return 1.0 if v else 0.0
        if isinstance(v, (int, float, np.number)):
            try: return float(v)
            except Exception: return 0.0
        if isinstance(v, str):
            try: return float(v)
            except Exception: return 0.0
        try: return float(v)
        except Exception: return 0.0
    row = {k: _cast(feat_map.get(k, 0.0)) for k in feature_order}
    return pd.DataFrame([row], columns=feature_order)

# ------------------- Normalización de entrada -------------------
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

# ----------------- Sidebar: ayuda + modo avanzado ----------------
with st.sidebar:
    st.header("¿Cómo usar?")
    st.markdown("1) Pegá la **URL completa** (ideal con `https://`).\n2) Tocá **Analizar**.\n3) Leé el nivel de riesgo.")
    DEV_MODE = st.toggle("Modo avanzado (técnico)", value=False)
    st.divider()
    st.subheader("Ejemplos")
    for e in ["https://www.afip.gob.ar/", "http://secure-login-afip.xyz/update", "https://www.bna.com.ar/"]:
        st.code(e, language="text")
    st.caption("Tip: ante dudas, escribí la dirección manualmente en el navegador.")

# ----------------- Entrada principal (usuario) ------------------
url_input = st.text_input("Pegá la URL a analizar", placeholder="https://www.ejemplo.com")
analizar = st.button("🔍 Analizar", use_container_width=True)

# ----------------- Carga de artefactos (silenciosa) -------------
try:
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    feature_order = load_feature_order(FEATURE_ORDER_PATH, model)
except Exception as e:
    st.error("No pudimos cargar el modelo. Volvé a intentar más tarde.")
    if DEV_MODE: st.exception(e)
    st.stop()

if scaler is None:
    st.error("Falta el archivo de escalado. Contactá al administrador.")
    st.stop()

# ----------------- Tacómetro con degradado ----------------------
def make_gradient_steps(n: int = 80, vmin: float = 0.0, vmax: float = 100.0):
    stops = [
        (0.00, (33, 197, 93)),   # #21c55d
        (0.50, (250, 204, 21)),  # #facc15
        (1.00, (239, 68, 68)),   # #ef4444
    ]
    def lerp(a, b, t): return a + (b - a) * t
    def interp_color(p: float):
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            if p <= p1:
                t = (p - p0) / (p1 - p0)
                r = int(lerp(c0[0], c1[0], t))
                g = int(lerp(c0[1], c1[1], t))
                b = int(lerp(c0[2], c1[2], t))
                return f"rgb({r},{g},{b})"
        r, g, b = stops[-1][1]
        return f"rgb({r},{g},{b})"
    steps = []
    span = vmax - vmin
    for i in range(n):
        a = vmin + (i * span) / n
        b = vmin + ((i + 1) * span) / n
        steps.append({"range": [a, b], "color": interp_color((i + 0.5) / n)})
    return steps

def render_tacometro_dashboard(prob: float, title: str = "Phishing Risk"):
    pct = round(prob * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 44, "color": "#101418", "family": "Arial Black"}},
            title={"text": "", "font": {"size": 1}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickmode": "array", "tickvals": [0,20,40,60,80,100],
                         "ticktext": ["0","20","40","60","80","100"], "tickwidth": 0, "ticks": ""},
                "bar": {"color": "rgba(0,0,0,0)"},
                "threshold": {"line": {"color": "#111", "width": 6}, "thickness": 0.9, "value": pct},
                "borderwidth": 0, "bgcolor": "rgba(0,0,0,0)",
                "steps": make_gradient_steps(n=80, vmin=0, vmax=100),
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=0), height=300,
        font=dict(color="#101418", family="Arial"), transition={"duration": 500, "easing": "cubic-in-out"},
    )
    st.markdown(f'<div class="gauge-card"><div class="gauge-title">{title}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="gauge-subtitle">Probabilidad estimada de phishing</div></div>', unsafe_allow_html=True)

# ------------------ Mapeo de riesgo a mensajes ------------------
def riesgo_bucket(p: float):
    if p < 0.20:   return "Bajo",  "Parece legítimo, pero mantené precaución."
    if p < 0.50:   return "Medio", "Verificá la dirección antes de interactuar."
    if p < 0.80:   return "Alto",  "No ingreses datos; comprobá por canales oficiales."
    return "Muy alto", "No hagas clic ni ingreses credenciales; reportalo."

# ------------------------ Predicción ----------------------------
def predict_and_show(dominio: str):
    try:
        with st.spinner("Analizando…"):
            base_feats = procesar_dominio_basico(dominio)
            dyn_feats  = enriquecer_dominio_scraping(dominio)
            feats = {**(base_feats or {}), **(dyn_feats or {})}

            expected_order = get_expected_order(load_feature_order(FEATURE_ORDER_PATH, model), scaler, model, list(feats.keys()))
            X = ensure_feature_vector(feats, expected_order)

            # preparar columnas para el scaler
            if hasattr(scaler, "feature_names_in_") and getattr(scaler, "feature_names_in_", None) is not None:
                scaler_cols = [str(c) for c in scaler.feature_names_in_]
            else:
                scaler_cols = [c for c in expected_order if c in NUMERIC_FEATURES]

            scaler_input = pd.DataFrame(columns=scaler_cols)
            for c in scaler_cols:
                scaler_input[c] = X[c].values if c in X.columns else 0.0

            scaled_array = scaler.transform(scaler_input[scaler_cols])

            # aplicar escalado solo a las columnas numéricas que coinciden
            overlap_cols = [c for c in scaler_cols if c in X.columns]
            if overlap_cols:
                idx_map = [scaler_cols.index(c) for c in overlap_cols]
                X.loc[:, overlap_cols] = scaled_array[:, idx_map]

            # predicción
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            else:
                proba = None
            y_pred = model.predict(X)

            label = int(y_pred[0]) if hasattr(y_pred, "__iter__") else int(y_pred)
            p_phishing = float(proba[0, 1]) if (proba is not None and np.ndim(proba) == 2 and proba.shape[1] >= 2) else (1.0 if label == 1 else 0.0)
    except Exception as e:
        st.error("No se pudo analizar la URL en este momento.")
        if DEV_MODE: st.exception(e)
        return

    # ---------- Presentación amigable ----------
    nivel, recomendacion = riesgo_bucket(p_phishing)
    render_tacometro_dashboard(p_phishing, title="Phishing Risk")

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown(f"**Resultado:** {'Posible PHISHING' if label == 1 else 'No phishing'}")
        st.markdown(f"**Probabilidad (clase 1):** {p_phishing:.3f}")
    with colB:
        st.markdown(f"**Nivel de riesgo:** {nivel}")
        st.caption(recomendacion)

    # Bloque de acciones rápidas (sin jerga)
    st.markdown("**Qué podés hacer ahora:**")
    st.markdown(
        "- No ingreses contraseñas ni datos bancarios.\n"
        "- Verificá la dirección escribiéndola manualmente en el navegador.\n"
        "- Si recibiste la URL por email/SMS, contactá al emisor por un **canal oficial**."
    )

    # ---------------- Solo si Modo avanzado ----------------
    if DEV_MODE:
        with st.expander("Detalles técnicos (solo avanzado)"):
            st.write("Dominio:", dominio)
            st.write("Vector de features (ordenado):")
            st.dataframe(X.T.rename(columns={0: "valor"}))

# ------------------------ Evento de UI --------------------------
if analizar:
    if not url_input.strip():
        st.warning("Ingresá una URL válida.")
    else:
        dominio = normalize_to_domain(url_input)
        if not dominio:
            st.warning("No se pudo interpretar la entrada como dominio/URL.")
        else:
            predict_and_show(dominio)

# ------------------- Descargo de responsabilidad ----------------
st.divider()
st.markdown(
    """
**Descargo de responsabilidad**  
Esta herramienta ofrece una **estimación automática** basada en un modelo de Machine Learning y **no garantiza** la legitimidad o ilicitud de un sitio.  
No reemplaza tu criterio ni verifica la identidad de ninguna organización.  
Ante dudas, **no ingreses datos sensibles** (contraseñas, tarjetas) y verificá por **canales oficiales**.
"""
)
st.caption('<span class="small-muted">Si necesitás un informe técnico, activá “Modo avanzado” en la barra lateral.</span>', unsafe_allow_html=True)
