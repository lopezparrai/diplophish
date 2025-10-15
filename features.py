# features.py
from __future__ import annotations

import re
import ssl
import time
import socket
import requests
import tldextract
import whois
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser
from urllib.parse import urlparse
from functools import lru_cache
from typing import Dict, Tuple, Optional

# ===== Config de red =====
DEFAULT_TIMEOUT = 6.0
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (phishing-checker)"}

def _safe_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    try:
        return requests.get(url, timeout=timeout, allow_redirects=True, headers=DEFAULT_HEADERS)
    except requests.exceptions.RequestException:
        return None

@lru_cache(maxsize=256)
def resolve_canonical_url(dominio: str) -> Tuple[Optional[str], int, bool, bool]:
    """
    Devuelve (final_url, status_code, has_https, responds).
    Unifica apex/www probando variantes y priorizando HTTPS; sigue redirects.
    """
    d = (dominio or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]

    candidates = [
        f"https://{d}",
        f"http://{d}",
        f"https://www.{d}",
        f"http://www.{d}",
    ]

    # 1) Primera 2xx/3xx
    for u in candidates:
        r = _safe_get(u)
        if r and (200 <= r.status_code < 400):
            final_url = str(r.url)
            return final_url, int(r.status_code), final_url.startswith("https://"), True

    # 2) Si nada 2xx/3xx, lo "mejor" que respondió
    best = None
    for u in candidates:
        r = _safe_get(u)
        if r and (best is None or r.status_code < best.status_code):
            best = r
    if best:
        final_url = str(best.url)
        return final_url, int(best.status_code), final_url.startswith("https://"), False

    return None, 0, False, False


# ===== Heurística general para path sospechoso =====
def is_suspicious_path(path: str, query: str) -> bool:
    """True si el path/query tiene señales típicas de phishing."""
    p = (path or "").lower()
    q = (query or "").lower()

    # 1) Palabras/tokens típicos
    bad_tokens = [
        "login","logon","signin","sign-in","acceso","ingreso",
        "verify","verification","verificacion","secure","security",
        "payment","pago","pagos","card","tarjeta","update","validate",
        "auth","otp","token","reset","recuperar","recovery"
    ]
    if any(t in p for t in bad_tokens) or any(t in q for t in bad_tokens):
        return True

    # 2) Estructura rara
    segs = [s for s in p.split("/") if s]
    if len(segs) >= 4:
        return True
    if any(len(s) >= 36 for s in segs):
        return True
    if "%" in p or "%" in q or "@/" in p or "@/" in q:
        return True
    if "//" in p:
        return True

    # 3) Extensiones de archivo sospechosas
    if segs and re.search(r"\.(exe|scr|bat|cmd|ps1|apk|jar|hta|vbs|js|zip|rar|7z)$", segs[-1]):
        return True

    # 4) Query con demasiados parámetros o base64-ish
    if q:
        if q.count("&") >= 5:
            return True
        if re.search(r"[A-Za-z0-9+/]{40,}={0,2}", q):
            return True

    return False


# ============================================================
# ===============  Features BASE (sin red)  ==================
# ============================================================
def procesar_dominio_basico(dominio: str) -> Dict[str, float]:
    """
    Procesa info estática y WHOIS.
    Las métricas sintácticas se calculan sobre el dominio registrable
    (registered_domain), NO sobre el host completo, para que 'www.' no
    infle el riesgo.
    """
    # 0) Resolver canónica para obtener host real (cacheado)
    final_url, _, _, _ = resolve_canonical_url(dominio)

    # 1) Host final
    if final_url:
        host_final = urlparse(final_url).hostname or (dominio or "").strip().lower()
    else:
        d = (dominio or "").strip().lower()
        if d.startswith(("http://", "https://")):
            d = urlparse(d).netloc or urlparse(d).path
        host_final = d[4:] if d.startswith("www.") else d

    # 2) Métricas sobre el REGISTRABLE (ej. 'afip.gob.ar')
    ext_final = tldextract.extract(host_final)
    registered = ext_final.registered_domain or host_final
    subdomain_depth = 0 if not ext_final.subdomain else len(ext_final.subdomain.split("."))

    # URL neutra para coherencia de longitud
    url = f"http://{registered}"
    parsed = urlparse(url)
    hostname = parsed.hostname or registered

    url_length = len(url)
    num_dashes = registered.count('-')
    num_digits = sum(c.isdigit() for c in registered)
    num_special_chars = len(re.findall(r'[^\w\s:/.-]', registered))
    num_dots = registered.count('.')                # ya no cuenta 'www.'
    num_underscores = registered.count('_')
    num_dashes_in_hostname = hostname.count('-')
    hostname_length = len(hostname)

    # En base no medimos path/query
    path_segments = 0
    path_length = 0
    query_length = 0
    double_slash_in_path = 0.0

    # TLD / WHOIS (usar el registrable)
    tld = tldextract.extract(registered).suffix

    creation_date_iso = None
    expiration_date_iso = None
    registrar = None
    country_registered = None
    site_age_years = None
    time_to_expire_years = None
    registration_time = None

    try:
        socket.setdefaulttimeout(5)
        info_whois = whois.whois(registered)

        creation = info_whois.creation_date
        expiration = info_whois.expiration_date

        if isinstance(creation, list) and creation:   creation = creation[0]
        if isinstance(expiration, list) and expiration: expiration = expiration[0]

        if isinstance(creation, str):
            try: creation = parser.parse(creation)
            except Exception: creation = None
        if isinstance(expiration, str):
            try: expiration = parser.parse(expiration)
            except Exception: expiration = None

        if creation:   creation_date_iso = creation.isoformat()
        if expiration: expiration_date_iso = expiration.isoformat()

        if creation:   site_age_years = round((datetime.now() - creation).days / 365, 2)
        if expiration: time_to_expire_years = round((expiration - datetime.now()).days / 365, 2)
        if creation and expiration:
            registration_time = round((expiration - creation).days / 365, 2)

        registrar = getattr(info_whois, "registrar", None)
        try:
            country_registered = (
                info_whois.get("country")
                or info_whois.get("registrant_country")
                or getattr(info_whois, "country", None)
                or "Desconocido"
            )
        except Exception:
            country_registered = getattr(info_whois, "country", "Desconocido")
    except Exception:
        pass

    # Fallbacks neutrales (evitar “edad 0” por WHOIS faltante)
    host_l = registered.lower()
    is_gov = host_l.endswith(".gob.ar") or ".gov." in host_l or host_l.endswith(".gov.ar")
    if site_age_years is None:       site_age_years = 8.0 if is_gov else 3.0
    if time_to_expire_years is None: time_to_expire_years = 1.0
    if registration_time is None:    registration_time = max(0.5, (time_to_expire_years + site_age_years) / 6.0)

    is_registered_in_ar = bool(country_registered and "argentina" in str(country_registered).lower())

    return {
        # Identificación
        "url": url,
        "tld": tld,

        # Estructura (sobre el registrable)
        "url_length": float(url_length),
        "num_dashes": float(num_dashes),
        "num_digits": float(num_digits),
        "num_special_chars": float(num_special_chars),
        "path_segments": float(path_segments),
        "num_dots": float(num_dots),
        "num_underscores": float(num_underscores),
        "num_dashes_in_hostname": float(num_dashes_in_hostname),
        "double_slash_in_path": float(double_slash_in_path),
        "hostname_length": float(hostname_length),
        "path_length": float(path_length),
        "query_length": float(query_length),

        # WHOIS / temporalidad
        "registration_time": float(registration_time),
        "creation_date": creation_date_iso,
        "expiration_date": expiration_date_iso,
        "site_age_years": float(site_age_years),
        "time_to_expire_years": float(time_to_expire_years),
        "registrar": registrar,
        "country_registered": country_registered,
        "is_registered_in_ar": 1.0 if is_registered_in_ar else 0.0,

        # Placeholders (se pisan en dinámicas)
        "title_length": 0.0,
        "http_status_code": 0.0,
        "has_https": 0.0,
        "has_ssl_cert": 0.0,
        "iframe_present": 0.0,
        "insecure_forms": 0.0,
        "submit_info_to_email": 0.0,
        "abnormal_form_action": 0.0,
        "responds": 0.0,
        "response_time": None,
        "sensitive_words_count": 0.0,

        # (opcional) profundidad de subdominio
        "subdomain_depth": float(subdomain_depth),
    }


# ============================================================
# ===========  Features DINÁMICAS (con red)  =================
# ============================================================
def enriquecer_dominio_scraping(dominio: str) -> Dict[str, float]:
    """
    Usa URL canónica (apex/www unificados) para medir señales de red/HTML.
    Además, el peso del path sólo se activa si hay señales objetivas de riesgo.
    """
    # 1) Resolver canónica y seguir redirects
    final_url, status_code, has_https, responds = resolve_canonical_url(dominio)

    titulo = ""
    tiempo_respuesta = None
    url_redireccionada = final_url
    tiene_ssl = False
    html_text = ""

    # 2) Descargar el HTML de la URL canónica (si existe)
    if final_url:
        t0 = time.time()
        r = _safe_get(final_url)
        t1 = time.time()
        if r is not None:
            tiempo_respuesta = round(t1 - t0, 3)
            html_text = r.text or ""
            status_code = int(r.status_code)
            url_redireccionada = str(r.url)

    # 3) Chequeo de SSL (sólo si la canónica es HTTPS)
    if final_url and has_https:
        try:
            host = urlparse(final_url).hostname
            if host:
                context = ssl.create_default_context()
                with socket.create_connection((host, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        tiene_ssl = bool(ssock.getpeercert())
        except Exception:
            pass

    # 4) Parse HTML
    soup = BeautifulSoup(html_text, "html.parser") if html_text else None
    if soup:
        t = soup.find("title")
        if t:
            titulo = (t.text or "").strip()

    longitud_titulo = float(len(titulo))
    parsed_final = urlparse(url_redireccionada or (final_url or ""))
    path = parsed_final.path or ""
    query = parsed_final.query or ""

    # --- Subdominio/Path robustos según HOST FINAL ---
    final_ext = tldextract.extract(parsed_final.netloc or "")
    final_reg = final_ext.registered_domain       # ej. "arca.gob.ar"
    final_sub = final_ext.subdomain               # ej. "", "www", "mi"
    domain_in_subdomains = 1.0 if final_sub not in ("", None, "www") else 0.0

    base_label = final_reg.split(".")[0] if final_reg else ""
    domain_in_paths = 1.0 if (base_label and base_label in (path or "")) else 0.0

    # --- Path "neutro" salvo que haya señales objetivas de riesgo ---
    if is_suspicious_path(path, query):
        path_segments_val = float(len([p for p in path.split("/") if p]))
        path_length_val   = float(len(path))
        query_length_val  = float(len(query))
    else:
        path_segments_val = 0.0
        path_length_val   = 0.0
        query_length_val  = 0.0

    # 5) Señales HTML y de formularios
    iframe_present = False
    insecure_forms = False
    submit_info_to_email = False
    abnormal_form_action = False

    if soup:
        iframe_present = bool(soup.find("iframe"))
        forms = soup.find_all("form")

        # Comparamos contra el REGISTRABLE del host final
        base_reg = final_reg
        for form in forms:
            action = (form.get("action") or "").strip().lower()

            # action vacío/fragmento: en HTTPS suele ser válido; en HTTP, inseguro
            if not action or action.startswith("#") or action.startswith("?"):
                if not has_https:
                    insecure_forms = True
                continue

            # action relativo: si el sitio es HTTPS, OK; si no, inseguro
            if not action.startswith(("http://", "https://")):
                if not has_https:
                    insecure_forms = True
                continue

            # absoluto http en sitio https → inseguro
            if has_https and action.startswith("http://"):
                insecure_forms = True

            # acción a otro registrable → abnormal, salvo cruces dentro de .gob.ar
            try:
                a_host = urlparse(action).hostname or ""
                a_reg = tldextract.extract(a_host).registered_domain
                if a_reg and base_reg and a_reg != base_reg:
                    if not (base_reg.endswith("gob.ar") and a_reg.endswith("gob.ar")):
                        abnormal_form_action = True
            except Exception:
                abnormal_form_action = True

    # 6) Indicadores engañosos (en el título / dominio)
    def _norm(s: str) -> str:
        s = (s or "").lower().strip()
        repl = (("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n"),("’","'"))
        for a,b in repl: s = s.replace(a,b)
        return re.sub(r"[^a-z0-9]", "", s)

    brands_raw = [
        "santander","bbva","galicia","banco nacion","bna","hipotecario","provincia","macro","comafi",
        "brubank","itau","supervielle","patagonia",
        "afip","anses","anmat","ministerio","gobierno","municipio","secretaria","renaper","dgr","dine",
        "senado","diputados","presidencia","arca",
        "facebook","instagram","twitter","whatsapp","tiktok","linkedin","telegram","snapchat",
        "mercadolibre","mercadopago","despegar","globant","tenaris","ypf","rappi","pedidosya","todopago",
        "naranja","uala","plin","cencosud"
    ]
    brands = {_norm(b) for b in brands_raw}

    dominio_norm = _norm(dominio)
    final_label_norm = _norm(final_reg.split(".")[0] if final_reg else "")
    embedded_brand_name = any(b in dominio_norm for b in brands)

    # Si la "marca" es exactamente el label del dominio oficial, NO penalizar
    if embedded_brand_name and final_label_norm in brands:
        embedded_brand_name = False

    https_in_hostname = 0.0
    random_string = 1.0 if re.search(r"[a-z]{5,}\d{3,}|[0-9]{5,}", (dominio or "").lower()) else 0.0

    categoria = clasificar_categoria(titulo or dominio)

    return {
        # Seguridad
        "has_https": 1.0 if has_https else 0.0,
        "has_ssl_cert": 1.0 if tiene_ssl else 0.0,
        "iframe_present": 1.0 if iframe_present else 0.0,
        "insecure_forms": 1.0 if insecure_forms else 0.0,
        "submit_info_to_email": 1.0 if submit_info_to_email else 0.0,
        "abnormal_form_action": 1.0 if abnormal_form_action else 0.0,

        # Respuesta servidor
        "response_time": float(tiempo_respuesta) if tiempo_respuesta is not None else None,
        "responds": 1.0 if responds else 0.0,
        "http_status_code": float(status_code or 0),
        "redirected_url": url_redireccionada,

        # Contenido
        "title": titulo,
        "title_length": longitud_titulo,
        "meta_keywords": "",
        "category": categoria,

        # Indicadores engañosos
        "random_string": float(random_string),
        "sensitive_words_count": float(sum((titulo or "").lower().count(w) for w in [
            "login","secure","account","bank","verify","update","contraseña","tarjeta","seguridad","verificación"
        ])),
        "embedded_brand_name": 1.0 if embedded_brand_name else 0.0,
        "https_in_hostname": float(https_in_hostname),
        "domain_in_subdomains": float(domain_in_subdomains),
        "domain_in_paths": float(domain_in_paths),

        # Longitudes derivadas de canónica (activas sólo si path sospechoso)
        "path_segments": path_segments_val,
        "path_length": path_length_val,
        "query_length": query_length_val,
        "double_slash_in_path": 1.0 if ("//" in (path or "") and not (path or "").startswith("//")) else 0.0,
    }


# ============================================================
# =============  Clasificación temática simple  ==============
# ============================================================
def clasificar_categoria(texto_o_dominio: str) -> str:
    """Clasifica en categorías amplias según palabras clave en título/dominio."""
    d = (texto_o_dominio or "").lower()

    if any(x in d for x in ["news","noticia","diario","prensa","periodico","press"]):
        return "noticias"
    if any(x in d for x in ["gob",".gob.","gov",".gov.","municipio","ministerio","provincia"]):
        return "gobierno"
    if any(x in d for x in ["banco","bank","finance","finanzas","credito","loan","tarjeta"]):
        return "banca"
    if any(x in d for x in ["shop","store","tienda","ecommerce","comprar","venta","oferta"]):
        return "e-commerce"
    if any(x in d for x in ["edu","universidad","facultad","campus","colegio","escuela","instituto"]):
        return "educacion"
    return "otro"
