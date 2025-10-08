
# streamlit_phishing_app_fixed.py
# ---------------------------------------------------------------
# Streamlit app to classify a URL as phishing / legit using a
# pre-trained model. Guarantees alignment with feature_order.json
# (22 features) via alias normalization + zero-filled defaults.
# ---------------------------------------------------------------
import json
import re
import socket
import ssl
import time
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

import streamlit as st
from sklearn.base import BaseEstimator
import joblib

# -------------------- Config --------------------
st.set_page_config(page_title="URL Phishing Classifier", layout="centered")

MODEL_PATH = Path("models/model_xgb.pkl")          # <-- ajusta si tu modelo tiene otro nombre
FEATURE_ORDER_PATH = Path("models/feature_order.json")  # Debe contener 22 columnas en JSON (lista)

# --------------- Utilidades varias ---------------
def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (bool, np.bool_)):
            return float(x)
        return float(x)
    except Exception:
        return float(default)

def _domain_from_url(url: str) -> Tuple[str, dict]:
    info = {"double_slash_in_path": 0.0, "has_https": 0.0}
    if not url:
        return "", info
    try:
        u = url.strip()
        if not re.match(r"^https?://", u, flags=re.I):
            u = "http://" + u  # normaliza mínimo
        parsed = urlparse(u)

        # Señal: // adicional en path (no el que sigue al esquema://)
        # Ej: http://example.com//login
        if parsed.path and '//' in parsed.path:
            info["double_slash_in_path"] = 1.0

        # Señal: usa https en el esquema
        info["has_https"] = 1.0 if parsed.scheme.lower() == "https" else 0.0
        return u, info
    except Exception:
        return url, info

# ----------------- Carga de artefactos -----------------
@st.cache_data(show_spinner=False)
def load_feature_order():
    with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_model() -> BaseEstimator:
    return joblib.load(MODEL_PATH)

# --------------- Alias -> nombre esperado ---------------
ALIAS_MAP = {
    # Conteos / longitudes
    "num_hyphens": "num_dashes",
    "hyphen_count": "num_dashes",
    "digits_count": "num_digits",
    "special_char_count": "num_special_chars",
    "hostname_dots": "num_dots",
    "dots_in_hostname": "num_dots",
    "len_hostname": "hostname_length",
    "len_path": "path_length",
    "len_query": "query_length",
    "underscores": "num_underscores",
    "dashes_in_host": "num_dashes_in_hostname",
    "url_len": "url_length",
    # HTTP / HTML / flags
    "status_code": "http_status_code",
    "https_in_url": "has_https",
    "has_ssl": "has_ssl_cert",
    "iframe": "iframe_present",
    "insecure_form": "insecure_forms",
    "submit_email": "submit_info_to_email",
    "abnormal_action": "abnormal_form_action",
    "double_slash_path": "double_slash_in_path",
    "registered_in_ar": "is_registered_in_ar",
    "alive": "responds",
    # Título
    "page_title_length": "title_length",
}

def normalize_feature_names(raw_feats: dict) -> dict:
    norm = {}
    for k, v in (raw_feats or {}).items():
        k2 = ALIAS_MAP.get(k, k)
        norm[k2] = _safe_float(v)
    return norm

# --------------- Extracción de features ---------------
def _count_chars(text: str, charset=r"[\-]", flags=0) -> int:
    if not text:
        return 0
    return len(re.findall(charset, text, flags))

def _get_hostname(parsed) -> str:
    host = parsed.hostname or ""
    return host

def extract_from_url_locally(url: str) -> Dict[str, float]:
    """
    Señales "baratas": longitudes, conteos en URL/host/path/query
    """
    feats = {}
    try:
        parsed = urlparse(url)
        host = _get_hostname(parsed)
        path = parsed.path or ""
        query = parsed.query or ""
        full  = url

        feats["url_length"] = len(full)
        feats["path_length"] = len(path)
        feats["query_length"] = len(query)
        feats["hostname_length"] = len(host)

        feats["num_dots"] = host.count(".")
        feats["num_dashes"] = _count_chars(full, r"[-]")
        feats["num_digits"] = _count_chars(full, r"[0-9]")
        feats["num_special_chars"] = _count_chars(full, r"[^A-Za-z0-9]")
        feats["num_underscores"] = _count_chars(full, r"[_]")
        feats["path_segments"] = len([p for p in path.split("/") if p])

        # Dashes en el hostname
        feats["num_dashes_in_hostname"] = host.count("-")
    except Exception:
        # si algo falla, se devuelven las que se hayan llenado
        pass
    return feats

def _try_tcp_connect(host: str, port: int, timeout=3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def _has_ssl_cert(host: str, timeout=3.0) -> float:
    """
    Hace un connect SSL y devuelve 1.0 si el handshake funciona.
    """
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                return 1.0 if cert else 0.0
    except Exception:
        return 0.0

def _is_registered_in_ar(host: str) -> float:
    """
    Heurística simple: TLD .ar
    """
    return 1.0 if host.lower().endswith(".ar") else 0.0

def enrich_with_network_signals(url: str) -> Dict[str, float]:
    """
    Señales que requieren realizar una request simple.
    """
    feats = {
        "http_status_code": 0.0,
        "responds": 0.0,
        "title_length": 0.0,
        "has_ssl_cert": 0.0,
    }
    try:
        parsed = urlparse(url)
        host = _get_hostname(parsed) or ""
        if not host:
            return feats

        # Intento de conexión (TCP) para "responds"
        if parsed.scheme == "https":
            ok = _try_tcp_connect(host, 443)
        else:
            ok = _try_tcp_connect(host, 80)
        feats["responds"] = 1.0 if ok else 0.0

        # SSL cert si es https
        if parsed.scheme == "https":
            feats["has_ssl_cert"] = _has_ssl_cert(host)

        # HTTP GET rápido (timeout bajo)
        try:
            resp = requests.get(url, timeout=4, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            feats["http_status_code"] = float(resp.status_code)

            # Intento de parseo HTML para título + flags de formulario/iframe
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.find("title")
                feats["title_length"] = float(len(title.text.strip())) if title and title.text else 0.0

                # iframes
                feats["iframe_present"] = 1.0 if soup.find("iframe") else 0.0

                # forms "inseguros": action con http sin https, o vacío
                insecure_forms = 0.0
                submit_info_to_email = 0.0
                abnormal_form_action = 0.0
                for form in soup.find_all("form"):
                    action = (form.get("action") or "").strip().lower()
                    if action.startswith("http://"):
                        insecure_forms = 1.0
                    # envío a email (mailto)
                    if action.startswith("mailto:") or "@" in action:
                        submit_info_to_email = 1.0
                    # acción sospechosa (vacía o a dominios no coincidentes)
                    if action == "" or action in ["/", "#"]:
                        abnormal_form_action = 1.0
                    else:
                        # dominio de action distinto al host
                        try:
                            ahost = urlparse(action).hostname or ""
                            if ahost and (ahost != host):
                                abnormal_form_action = 1.0
                        except Exception:
                            pass
                feats["insecure_forms"] = insecure_forms
                feats["submit_info_to_email"] = submit_info_to_email
                feats["abnormal_form_action"] = abnormal_form_action
            except Exception:
                pass
        except Exception:
            # request falló: deja status_code=0, title_length=0 y resto como están
            pass
    except Exception:
        pass
    return feats

def enrich_with_whois_and_ccTLD(url: str) -> Dict[str, float]:
    """
    Para mantenerlo sin dependencias extra, solo devolvemos is_registered_in_ar
    basado en TLD. Si usas python-whois, puedes mejorar este bloque.
    """
    feats = {"is_registered_in_ar": 0.0}
    try:
        parsed = urlparse(url)
        host = _get_hostname(parsed) or ""
        feats["is_registered_in_ar"] = _is_registered_in_ar(host)
    except Exception:
        pass
    return feats

# ----------- Orquestación: 22 features garantizadas -----------
def build_feature_vector(url: str, feature_order) -> Dict[str, float]:
    # Inicializa exactamente con las 22 (o las que defina el JSON)
    feats = {k: 0.0 for k in feature_order}

    # Normaliza URL y señales inmediatas
    url_norm, base_flags = _domain_from_url(url)

    # Señales locales (baratas)
    f_local = extract_from_url_locally(url_norm)
    # Señales de red / HTML
    f_net   = enrich_with_network_signals(url_norm)
    # WHOIS / ccTLD
    f_who   = enrich_with_whois_and_ccTLD(url_norm)

    merged = {}
    for d in (base_flags, f_local, f_net, f_who):
        merged.update(d)

    merged_norm = normalize_feature_names(merged)

    # Pisar los 22 ceros con lo que tengamos
    for k in feats.keys():
        if k in merged_norm:
            feats[k] = _safe_float(merged_norm[k], 0.0)

    return feats

def vectorize_in_order(feats: Dict[str, float], feature_order):
    return np.array([[feats[k] for k in feature_order]], dtype=float)

def predict_proba_single(model: BaseEstimator, url: str, feature_order) -> Tuple[float, Dict[str, float]]:
    feats = build_feature_vector(url, feature_order)
    x = vectorize_in_order(feats, feature_order)
    proba = float(model.predict_proba(x)[0, 1])
    return proba, feats

# ---------------------- UI ----------------------
st.title("🔎 URL Phishing Classifier")
st.caption("Modelo pre-entrenado · Vector de 22 features garantizado")

with st.sidebar:
    st.subheader("⚙️ Configuración")
    model_path = st.text_input("Ruta del modelo (.pkl)", value=str(MODEL_PATH))
    feat_path  = st.text_input("Ruta de feature_order.json", value=str(FEATURE_ORDER_PATH))
    threshold  = st.slider("Umbral de clasificación (phishing si proba ≥ umbral)", 0.0, 1.0, 0.5, 0.01)

# Reemplaza rutas si el usuario las cambia
MODEL_PATH = Path(model_path)
FEATURE_ORDER_PATH = Path(feat_path)

# Carga artefactos (con manejo de errores amigable)
try:
    feature_order = load_feature_order()
except Exception as e:
    st.error(f"No pude cargar feature_order.json desde {FEATURE_ORDER_PATH} → {e}")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.warning(f"No pude cargar el modelo desde {MODEL_PATH}. "
               f"Puedes seguir probando la extracción de features, "
               f"pero no habrá predicción. Detalle: {e}")
    model = None

url_input = st.text_input("Ingresa una URL para analizar", placeholder="https://www.ejemplo.com/login")

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    run = st.button("Analizar", type="primary")
with col_btn2:
    demo = st.button("Probar con ejemplo", help="Carga una URL de prueba")

if demo and not url_input:
    url_input = "https://www.afip.gob.ar/"
    st.info(f"URL de ejemplo: {url_input}")

if run or demo or url_input:
    if not url_input:
        st.stop()

    with st.spinner("Extrayendo features..."):
        try:
            proba, feats = predict_proba_single(model, url_input, feature_order) if model else (np.nan, build_feature_vector(url_input, feature_order))
        except Exception as e:
            st.error(f"Error durante la extracción/predicción: {e}")
            st.stop()

    # Resultado
    if model:
        label = "Phishing" if proba >= threshold else "Legítima"
        color = "🛑" if label == "Phishing" else "✅"
        st.subheader(f"Resultado: {color} {label}")
        st.write(f"**Probabilidad (clase phishing)**: `{proba:.4f}` — **Umbral**: `{threshold:.2f}`")
    else:
        st.info("Modelo no cargado: se muestran únicamente las features extraídas.")

    # Expander de features
    with st.expander("Ver features extraídos (alineados con el modelo)"):
        st.write(f"Total esperadas por el modelo: **{len(feature_order)}**")
        st.write(f"Total presentes tras normalización: **{len(feats)}**")

        # Lista ordenada en el mismo orden del modelo
        rows = [(k, feats.get(k, 0.0)) for k in feature_order]
        st.dataframe({"feature": [k for k,_ in rows], "value": [v for _,v in rows]}, hide_index=True)

    # Expander de diferencias nominales (por si el extractor original trae otros nombres)
    with st.expander("Ver diferencias de columnas (esperadas vs presentes en dict)"):
        present_keys = set(feats.keys())
        expected_set = set(feature_order)

        missing = [k for k in feature_order if k not in present_keys]
        extras  = [k for k in present_keys if k not in expected_set]

        st.write(f"**Features esperadas**: {len(feature_order)} — **Presentes**: {len(present_keys)}")
        st.write(f"Faltan ({len(missing)}): {', '.join(missing) if missing else '—'}")
        st.write(f"Sobran ({len(extras)}): {', '.join(extras) if extras else '—'}")

    st.caption(
        f"Modelo: {MODEL_PATH.name if model else '—'} · Features usados: {len(feature_order)} · Umbral: {threshold:.2f}"
    )
