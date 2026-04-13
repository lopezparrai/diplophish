# 🛡️ DiploPhish — Detector de URLs de Phishing

Proyecto de Mentoría desarrollado en el marco de la **Diplomatura en Ciencia de Datos, ML y sus Aplicaciones** (FAMAF – UNC, 2025).

**Grupo 1:** Manuel Lopez Werlen · Ayelen Margarita Bertorello · Silvio Fabian Marasca · Ignacio Ariel Lopez Parra  
**Mentora:** Noelia Ferrero

El objetivo fue construir un **pipeline completo de Machine Learning** para clasificar URLs como phishing o legítimas, con foco en dominios argentinos (`.ar`). El resultado final es una aplicación web interactiva construida con Streamlit que permite analizar cualquier URL en tiempo real.

---

## 📌 Descripción del problema

El phishing es una de las amenazas de ciberseguridad más frecuentes en Argentina, especialmente dirigida a usuarios de entidades bancarias, organismos públicos (AFIP, ANSES, PAMI, ARBA, BCRA) y plataformas de e-commerce. Este proyecto aborda el problema desde una perspectiva de Ciencia de Datos: extrayendo features estructurales, de red y de contenido a partir de una URL, y entrenando un modelo de clasificación basado en **XGBoost**.

---

## 🗂️ Estructura del repositorio

```
diplophish/
├── notebooks/
│   ├── 01_recoleccion_y_eda.ipynb                   # Entregable 1: recolección, EDA, scraping
│   ├── 02_curacion_y_feature_engineering.ipynb       # Entregable 2: limpieza, correlaciones, dataset maestro
│   └── 03_modelado_supervisado.ipynb                 # Entregable 3: modelos, comparación, SHAP
├── models/
│   ├── xgb_phishing.pkl                              # Modelo XGBoost entrenado
│   └── standard_scaler.pkl                           # Scaler ajustado
├── features.py                                       # Extracción de features (WHOIS + scraping)
├── streamlit_phishing_app.py                         # App de deployment
├── streamlit_phishing_app2.py                        # Versión alternativa de la app
├── requirements.txt
└── README.md
```

---

## 🏗️ Pipeline completo

```
Fuentes de datos (Tranco List + OpenPhish + PhishTank + CT Logs)
     │
     ▼
┌──────────────────────────────────┐
│  Notebook 01 — Recolección y EDA │
│  ├── Descarga dominios .ar       │  ← Tranco List top 1M
│  ├── Extracción WHOIS + scraping │  ← features estructurales y de red
│  ├── Recolección phishing        │  ← OpenPhish (URLs reales en circulación)
│  ├── Datos sintéticos            │  ← Faker para augmentación
│  └── EDA: patrones sospechosos   │  ← TLD, SSL, fechas, tiempos de respuesta
└────────────────┬─────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Notebook 02 — Curación y Feature Eng. │
│  ├── Tratamiento de nulos y duplicados │
│  ├── Corrección de tipos de datos      │
│  ├── Balance de clases (2 estrategias) │  ← con/sin datos sintéticos
│  ├── StandardScaler                    │
│  ├── Análisis de correlaciones         │  ← heatmap + relación con is_phishing
│  └── Dataset maestro final             │  ← master_dataset2.csv
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Notebook 03 — Modelado Supervisado     │
│  ├── Regresión Logística (baseline)     │
│  ├── Árbol de Decisión                  │
│  ├── Random Forest                      │
│  ├── Random Forest + SMOTE              │
│  ├── XGBoost (baseline)                 │
│  ├── XGBoost + GridSearchCV ✅ elegido  │
│  ├── Análisis SHAP                      │
│  └── Feature importance + comparación  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌──────────────────────────┐
│  Deployment — Streamlit  │  ← predicción en tiempo real para cualquier URL
└──────────────────────────┘
```

---

## 📓 Notebooks

### [01 — Recolección y EDA](notebooks/01_recoleccion_y_eda.ipynb)

**Objetivo:** construir el dataset inicial y realizar un primer análisis exploratorio.

**Fuentes de datos:**
- **Sitios legítimos:** top 1.000.000 de Tranco List, filtrado a dominios `.ar` (~3.200 dominios)
- **Sitios phishing:** feed público de OpenPhish (URLs maliciosas en circulación real)

**Procesamiento implementado:**
- `procesar_dominio_basico()`: extrae features estructurales de la URL y datos WHOIS (antigüedad, registrar, país, fechas de creación y vencimiento)
- `enriquecer_dominio_scraping()`: realiza requests HTTP/HTTPS para obtener título, tiempo de respuesta, iframes, formularios inseguros, redirecciones y código de estado HTTP
- Clasificación temática de dominios por palabras clave (banca, gobierno, noticias, e-commerce, etc.)
- Generación de datos sintéticos con Faker para aumentar la clase phishing

**Hallazgos del EDA:**
- Solo el **18%** de los sitios legítimos `.ar` tiene TLDs distintos de `.com.ar`, `.edu.ar` o `.gob.ar` — los TLDs inusuales son señal de sospecha
- La mayoría de los sitios legítimos fueron creados entre **2010 y 2015**; ninguno en los últimos 10 años, consistente con su popularidad consolidada
- Casi el **80%** de los sitios legítimos usa SSL — la ausencia de SSL es indicador de alerta
- La mayoría de los sitios responde en **menos de 1 segundo**; tiempos altos pueden indicar comportamiento sospechoso

**Outputs generados:**
- `sitios_argentinos_procesados.csv`
- `sitios_fraudulentos.csv`
- `sitios_argentinos_enriquecidos.csv`
- `sitios_argentinos_sinteticos.csv`

---

### [02 — Curación y Feature Engineering](notebooks/02_curacion_y_feature_engineering.ipynb)

**Objetivo:** limpiar, curar y preparar el dataset maestro listo para modelado.

**Composición del dataset:**

| Tipo de sitio | Fuente | Registros | `is_phishing` |
|---|---|---|---|
| Legítimos | Tranco List + CT Logs + Scraping | 5.758 | False |
| Phishing reales | PhishTank, PhishStats, OpenPhish, URLhaus | 1.000+ | True |
| Phishing + sintéticos | Reales + Faker | aumentado | True |

Se construyeron dos datasets maestros: `dataMaestro1` (sin datos sintéticos) y `dataMaestro2` (con datos sintéticos), comparando su impacto en el modelado posterior.

**Tratamiento aplicado:**
- **Nulos:** 12 de 43 columnas tenían valores faltantes. `meta_keywords` eliminada (85.7% nulos). Fechas imputadas con mediana. Variables numéricas restantes imputadas a 0.
- **Duplicados:** eliminados por columna `url`, conservando primera ocurrencia
- **Tipos:** conversión de fechas a `datetime`, `http_status_code` a `Int64`
- **Escalado:** `StandardScaler` aplicado sobre variables numéricas seleccionadas

**Análisis de correlaciones:**
- Variables de estructura de URL (`url_length`, `path_length`, `path_segments`, `num_dots`) tienen alta correlación entre sí por naturaleza
- Variables **más correlacionadas con `is_phishing`**: `path_segments`, `url_length`, `path_length`, `num_dots` — los sitios phishing tienden a URLs más largas y complejas
- Correlaciones moderadas con el target: `site_age_years` (phishing son más recientes) y `has_ssl_cert`

**Output:** `master_dataset2.csv` — dataset maestro escalado y listo para modelado

---

### [03 — Modelado Supervisado](notebooks/03_modelado_supervisado.ipynb)

**Objetivo:** entrenar, comparar y seleccionar el mejor clasificador binario para detección de phishing.

**División del dataset:** 70% train / 15% validación / 15% test, con estratificación por clase.

**Modelos evaluados:**

| Modelo | Descripción | Resultado |
|---|---|---|
| Regresión Logística | Baseline simple sin ajustes | Precisión perfecta pero recall 83.6% — deja escapar 1 de cada 6 phishing |
| Árbol de Decisión | Clasificador único sin ensemble | Recall apenas mejora, precisión cae ~13% |
| Random Forest | Ensemble de árboles con votación por mayoría | Recupera alta precisión, PR-AUC = 0.927 |
| Random Forest + SMOTE | Random Forest con oversampling sintético | Recall sube a 0.877 pero precisión cae significativamente |
| XGBoost (base) | Gradient Boosting secuencial, parámetros por defecto | Resultados sólidos como punto de partida |
| **XGBoost + GridSearchCV** ✅ | XGBoost con `scale_pos_weight` + búsqueda de hiperparámetros optimizando PR-AUC | **Modelo final seleccionado** |

**Hiperparámetros ajustados en GridSearch:** número de árboles, profundidad máxima, tasa de aprendizaje, `scale_pos_weight` para manejo de desbalance de clases.

**Interpretabilidad:**
- **Análisis SHAP** (`TreeExplainer`): valores SHAP globales e individuales para entender cómo cada feature impacta en cada predicción
- **Feature importance** de XGBoost: las variables de estructura de URL y comportamiento HTTP son las más relevantes
- **Correlación con target**: variables con correlación > 0.7 analizadas para descartar leakage

**Conclusión:** XGBoost con ajuste de hiperparámetros logra el mejor equilibrio entre precisión y recall, siendo el modelo más adecuado para el deployment en contextos con clases desbalanceadas.

---

## 🔍 Features extraídas

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
| `title_length` | Longitud del título |

---

## 🖥️ Aplicación web (Streamlit)

La app permite ingresar cualquier URL o dominio y obtener:
- **Predicción** del modelo (Legítimo / Phishing)
- **Probabilidad estimada** de phishing
- **Vector de features** utilizado para la predicción

```bash
streamlit run streamlit_phishing_app.py
```

La app estará disponible en `http://localhost:8501`.

---

## 🚀 Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/lopezparrai/diplophish.git
cd diplophish

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run streamlit_phishing_app.py
```

> Asegurate de tener los archivos `models/xgb_phishing.pkl` y `models/standard_scaler.pkl` en la carpeta `models/`.

---

## 📦 Dependencias principales

```
streamlit
xgboost
scikit-learn
imbalanced-learn
shap
pandas
numpy
requests
beautifulsoup4
tldextract
python-whois
python-dateutil
joblib
missingno
faker
fake_useragent
```

---

## 🎓 Contexto académico

Proyecto de Mentoría de la [Diplomatura en Ciencia de Datos, ML y sus Aplicaciones](https://diplodatos.famaf.unc.edu.ar/) — FAMAF, Universidad Nacional de Córdoba (2025).

---

## ⚠️ Disclaimer

Este clasificador es un prototipo con fines académicos. **No debe usarse como única herramienta de seguridad** en entornos productivos. La detección de phishing es un problema dinámico y los modelos requieren reentrenamiento periódico para mantenerse efectivos.
