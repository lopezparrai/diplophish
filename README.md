# 🛡️ DiploPhish — Detector de URLs de Phishing

Proyecto de Mentoría desarrollado en el marco de la **Diplomatura en Ciencia de Datos, ML y sus Aplicaciones** (FAMAF – UNC).

El objetivo fue construir un pipeline completo de Machine Learning para clasificar URLs como **phishing** o **legítimas**, con foco en dominios argentinos. El resultado final es una aplicación web interactiva construida con Streamlit que permite analizar cualquier URL en tiempo real.

---

## 📌 Descripción

El phishing es una de las amenazas de ciberseguridad más frecuentes en Argentina, especialmente dirigida a usuarios de entidades bancarias, organismos públicos y plataformas de e-commerce. Este proyecto aborda el problema desde una perspectiva de Ciencia de Datos: extrayendo features estructurales, de red y de contenido a partir de una URL, y entrenando un modelo de clasificación basado en **XGBoost**.

El pipeline cubre desde la recolección y etiquetado de datos hasta el deployment de la aplicación.

---

## 🏗️ Arquitectura del pipeline

```
URL / Dominio
     │
     ▼
┌─────────────────────────┐
│  Extracción de features │
│  ├── features básicas   │  ← procesar_dominio_basico()
│  │    (URL, WHOIS)      │
│  └── features dinámicas │  ← enriquecer_dominio_scraping()
│       (HTTP, HTML)      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Preprocesamiento       │
│  ├── Imputación (→ 0)   │
│  └── StandardScaler     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Modelo XGBoost         │
│  (clasificación binaria)│
└──────────┬──────────────┘
           │
           ▼
   Predicción + Probabilidad
   (0 = Legítimo / 1 = Phishing)
```

---

## 🔍 Features extraídas

El módulo `features.py` extrae dos grupos de características:

### Estructura de la URL
| Feature | Descripción |
|---|---|
| `url_length` | Longitud total de la URL |
| `num_dashes` | Cantidad de guiones en el dominio |
| `num_digits` | Cantidad de dígitos |
| `num_special_chars` | Caracteres especiales |
| `num_dots` | Cantidad de puntos |
| `num_underscores` | Cantidad de guiones bajos |
| `num_dashes_in_hostname` | Guiones en el hostname |
| `hostname_length` | Longitud del hostname |
| `path_length` | Longitud del path |
| `query_length` | Longitud del query string |
| `path_segments` | Segmentos del path |
| `double_slash_in_path` | Doble barra en el path |
| `tld` | TLD del dominio |

### WHOIS y temporalidad
| Feature | Descripción |
|---|---|
| `site_age_years` | Antigüedad del dominio en años |
| `time_to_expire_years` | Tiempo hasta el vencimiento |
| `registration_time` | Duración total del registro |
| `creation_date` | Fecha de creación |
| `expiration_date` | Fecha de vencimiento |
| `registrar` | Registrador del dominio |
| `country_registered` | País de registro |
| `is_registered_in_ar` | Flag: registrado en Argentina |

### Seguridad y comportamiento HTTP
| Feature | Descripción |
|---|---|
| `has_https` | Usa HTTPS |
| `has_ssl_cert` | Tiene certificado SSL válido |
| `iframe_present` | Contiene iframes |
| `insecure_forms` | Formularios con acción insegura (HTTP) |
| `submit_info_to_email` | Formularios que envían a email |
| `abnormal_form_action` | Formularios que apuntan a otro dominio |
| `response_time` | Tiempo de respuesta del servidor |
| `responds` | El dominio responde |
| `http_status_code` | Código de estado HTTP |
| `redirected_url` | URL final tras redirecciones |

### Indicadores de engaño (heurísticos)
| Feature | Descripción |
|---|---|
| `sensitive_words_count` | Palabras clave sensibles en el título (login, bank, verify…) |
| `embedded_brand_name` | Contiene nombre de marca conocida en el dominio |
| `https_in_hostname` | La palabra "https" aparece dentro del hostname |
| `random_string` | Patrón de string aleatorio en el dominio |
| `domain_in_subdomains` | Dominio base aparece como subdominio |
| `domain_in_paths` | Dominio base aparece en el path |
| `category` | Categoría temática (banca, gobierno, noticias…) |
| `title` | Título de la página |
| `title_length` | Longitud del título |
| `meta_keywords` | Meta keywords del HTML |

---

## 🖥️ Aplicación web (Streamlit)

La app permite ingresar cualquier URL o dominio y obtener la predicción del modelo en tiempo real, junto con la probabilidad estimada y el vector de features utilizado.

> 📸 **Capturas de pantalla**
>
> *(Agregar imágenes aquí — podés arrastrarlas a la carpeta `assets/` y referenciarlas así:)*
>
> ```markdown
> ![Pantalla principal](assets/screenshot_main.png)
> ![Resultado phishing](assets/screenshot_phishing.png)
> ![Resultado legítimo](assets/screenshot_legitimo.png)
> ```

---

## 🚀 Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/lopezparrai/diplophish.git
cd diplophish
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar estructura de modelos

Asegurate de tener los archivos entrenados en la carpeta `models/`:

```
diplophish/
├── models/
│   ├── xgb_phishing.pkl        # Modelo XGBoost entrenado
│   └── standard_scaler.pkl     # Scaler ajustado
├── feature_order.json          # Orden de features (opcional)
├── features.py
├── streamlit_phishing_app.py
└── requirements.txt
```

### 5. Ejecutar la aplicación

```bash
streamlit run streamlit_phishing_app.py
```

La app estará disponible en `http://localhost:8501`.

---

## 📦 Dependencias principales

```
streamlit
xgboost
scikit-learn
pandas
numpy
requests
beautifulsoup4
tldextract
python-whois
python-dateutil
joblib
```

---

## 📁 Estructura del proyecto

```
diplophish/
├── models/                         # Modelos entrenados (no incluidos en el repo)
│   ├── xgb_phishing.pkl
│   └── standard_scaler.pkl
├── features.py                     # Extracción de features (WHOIS + scraping)
├── streamlit_phishing_app.py       # App de deployment (con scaler)
├── streamlit_phishing_app2.py      # Versión alternativa de la app
├── requirements.txt                # Dependencias
└── README.md
```

---

## 🎓 Contexto académico

Este proyecto fue desarrollado como **Proyecto de Mentoría** de la [Diplomatura en Ciencia de Datos, ML y sus Aplicaciones](https://diplodatos.famaf.unc.edu.ar/) de FAMAF (Universidad Nacional de Córdoba).

---

## ⚠️ Disclaimer

Este clasificador es un prototipo con fines académicos. **No debe usarse como única herramienta de seguridad** en entornos productivos. La detección de phishing es un problema dinámico y los modelos requieren reentrenamiento periódico para mantenerse efectivos.
