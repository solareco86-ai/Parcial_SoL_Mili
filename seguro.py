import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

st.set_page_config(page_title='Agente IA - Accidentes Personales', layout='centered')

# ----------------------------
# Lectura robusta de CSV
# ----------------------------
def read_csv_flexible(path):
    for enc in ["utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    st.error(f"No se pudo leer {path}. Revisá el archivo o convertí a CSV UTF-8.")
    return pd.DataFrame()

# ----------------------------
# Dataset de entrenamiento
# ----------------------------
@st.cache_data
def cargar_dataset(path='dataset_accidentes.csv'):
    if os.path.exists(path):
        df = read_csv_flexible(path)
        df.columns = [c.lower() for c in df.columns]
        return df

    # Sintético si no hay dataset
    np.random.seed(42)
    n = 2000
    edad = np.random.randint(18, 70, n)
    peso = np.random.normal(75, 12, n).clip(45, 140).round(1)
    talla = np.random.normal(170, 8, n).clip(140, 200).round(1)
    fumador = np.random.choice([0, 1], n, p=[0.75, 0.25])
    actividad = np.random.choice(['Oficina', 'Industria', 'Construccion', 'Transporte'], n)
    imc = (peso / ((talla/100)**2)).round(1)
    riesgo_sint = ((edad > 55).astype(int) + (fumador == 1).astype(int) + (imc > 30).astype(int))
    asegurable = (riesgo_sint <= 1).astype(int)

    return pd.DataFrame({
        'edad': edad, 'peso': peso, 'talla': talla,
        'imc': imc, 'fumador': fumador,
        'actividad': actividad, 'asegurable': asegurable
    })

df = cargar_dataset()

# ----------------------------
# Entrenamiento
# ----------------------------
FEATURES = ['edad', 'peso', 'talla', 'fumador', 'actividad']
TARGET = 'asegurable'

@st.cache_data
def entrenar_modelo(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        OHE = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except:
        OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preproc = ColumnTransformer([
        ('num', StandardScaler(), ['edad', 'peso', 'talla']),
        ('cat', OHE, ['actividad', 'fumador'])
    ])

    clf = Pipeline([
        ('pre', preproc),
        ('rf', RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    joblib.dump(clf, 'model_accidentes_v1.joblib')
    return clf, metrics

modelo, metrics = entrenar_modelo(df)

# ----------------------------
# Extracción PDF mejorada
# ----------------------------
# ----------------------------
# Extracción PDF mejorada con soporte para "NO"
# ----------------------------
def extraer_pdf(file_like):
    texto = ''
    try:
        with pdfplumber.open(file_like) as pdf:
            for p in pdf.pages:
                texto += (p.extract_text() or '') + '\n'
                try:
                    for table in p.extract_tables():
                        for row in table:
                            texto += ' '.join([str(c) for c in row if c]) + '\n'
                except:
                    pass
    except Exception as e:
        return {'error': str(e)}

    valores, txt = {}, texto.lower()

    def buscar_regex(pattern, tipo=str):
        m = re.search(pattern, txt)
        if not m:
            return None
        raw = m.group(1)
        if tipo is float:
            return float(str(raw).replace(",", "."))
        if tipo is int:
            return int(re.sub(r"[^\d]", "", raw))
        return raw.strip()

    # Variables básicas
    valores['edad'] = buscar_regex(r'edad[:\s]*([0-9]{1,3})', int)
    peso = buscar_regex(r'peso[:\s]*([0-9]{2,3}(?:[\.,][0-9]+)?)', float)
    if peso is not None:
        valores['peso'] = peso
    talla = buscar_regex(r'(?:talla|altura)[:\s]*([0-9]{2,3}(?:[\.,][0-9]+)?)', float)
    if talla is not None:
        valores['talla'] = talla

    # Sexo
    sexo = buscar_regex(r'sexo[:\s]*(hombre|mujer|masculino|femenino|m|f)')
    if sexo:
        valores['sexo'] = "M" if sexo[0] in ['h', 'm'] else "F"

    # Función auxiliar para detectar "NO"
    def flag_negativo(pat):
        if re.search(pat + r'.{0,5}\s*no', txt):  # detecta "Diabetes No", "Hipertensión: No"
            return 0
        elif re.search(pat, txt):
            return 1
        return 0

    # Condiciones de salud con soporte para "NO"
    valores['diabetes']      = flag_negativo(r'diabetes')
    valores['hipertension']  = flag_negativo(r'hipertensi[oó]n')
    valores['enf_renal']     = flag_negativo(r'(enfermedad renal|insuficiencia renal)')
    valores['asma']          = flag_negativo(r'(asma|respiratoria)')
    valores['no_apto']       = 1 if re.search(r'\bno\s*apto\b', txt) else 0

    # IMC
    if 'peso' in valores and 'talla' in valores:
        talla_m = valores['talla']/100 if valores['talla'] > 3 else valores['talla']
        valores['imc'] = round(valores['peso'] / (talla_m**2), 2)

    return valores

    def buscar_regex(pattern, tipo=str):
        m = re.search(pattern, txt)
        if not m:
            return None
        raw = m.group(1)
        if tipo is float:
            return float(str(raw).replace(",", "."))
        if tipo is int:
            return int(re.sub(r"[^\d]", "", raw))
        return raw.strip()

    # Variables básicas
    valores['edad'] = buscar_regex(r'edad[:\s]*([0-9]{1,3})', int)
    peso = buscar_regex(r'peso[:\s]*([0-9]{2,3}(?:[\.,][0-9]+)?)', float)
    if peso is not None:
        valores['peso'] = peso
    talla = buscar_regex(r'(?:talla|altura)[:\s]*([0-9]{2,3}(?:[\.,][0-9]+)?)', float)
    if talla is not None:
        valores['talla'] = talla

    # Sexo
    sexo = buscar_regex(r'sexo[:\s]*(hombre|mujer|masculino|femenino|m|f)')
    if sexo:
        valores['sexo'] = "M" if sexo[0] in ['h', 'm'] else "F"

    # Fumador (soporta frases tipo "no fumador")
    if re.search(r'no\s+fumador', txt):
        valores['fumador'] = 0
    else:
        fum = buscar_regex(r'fumador[:\s]*(sí|si|no)')
        if fum:
            valores['fumador'] = 1 if fum in ['sí', 'si'] else 0

    # Condiciones de salud / marca NO APTO
    valores['diabetes']      = 1 if re.search(r'diabetes', txt) else 0
    valores['hipertension']  = 1 if re.search(r'hipertensi[oó]n', txt) else 0
    valores['enf_renal']     = 1 if re.search(r'enfermedad renal|insuficiencia renal', txt) else 0
    valores['asma']          = 1 if re.search(r'asma|respiratoria', txt) else 0
    valores['no_apto']       = 1 if re.search(r'\bno\s*apto\b', txt) else 0

    # IMC
    if 'peso' in valores and 'talla' in valores:
        talla_m = valores['talla']/100 if valores['talla'] > 3 else valores['talla']
        valores['imc'] = round(valores['peso'] / (talla_m**2), 2)

    return valores

# ----------------------------
# Tablas de primas
# ----------------------------
tarifa_enf_df = read_csv_flexible("tarifa_seguros_enfermedades.csv")
tarifa_salud_df = read_csv_flexible("tarifa_salud_argentina.csv")
tarifa_base_df = read_csv_flexible("tarifa_seguros.csv")

for df_tmp in [tarifa_enf_df, tarifa_salud_df, tarifa_base_df]:
    if not df_tmp.empty:
        df_tmp.columns = [c.lower() for c in df_tmp.columns]

if tarifa_enf_df.empty:
    tarifa_enf_df = pd.DataFrame(columns=['edad','sexo','fumador','enfermedad','factor'])
if tarifa_salud_df.empty:
    tarifa_salud_df = pd.DataFrame(columns=['edad','sexo','fumador','enfermedad','factor'])
if tarifa_base_df.empty:
    tarifa_base_df = pd.DataFrame(columns=['edad','sexo','fumador','prima_base'])

# ----------------------------
# Cálculo de prima
# ----------------------------
GASTOS_Y_UTILIDAD = 0.25
IMP_TASAS_SELLOS = 0.05

def buscar_prima(df, edad, sexo, fumador, enfermedad=None):
    if df is None or df.empty:
        return None
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    sexo = str(sexo or '').strip().upper()
    fum_str = 'sí' if int(fumador or 0) == 1 else 'no'

    if 'sexo' in d.columns:
        d['sexo'] = d['sexo'].astype(str).str.strip().str.upper()
        if sexo:
            d = d[d['sexo'] == sexo]
    if 'fumador' in d.columns:
        d['fumador'] = d['fumador'].astype(str).str.strip().str.lower().replace({'si':'sí'})
        d = d[d['fumador'] == fum_str]
    if enfermedad and 'enfermedad' in d.columns:
        d['enfermedad'] = d['enfermedad'].astype(str).str.strip().str.lower()
        d = d[d['enfermedad'] == enfermedad.lower()]

    if d.empty:
        return None

    if 'edad_min' in d.columns and 'edad_max' in d.columns:
        d['edad_min'] = pd.to_numeric(d['edad_min'], errors='coerce').fillna(0).astype(int)
        d['edad_max'] = pd.to_numeric(d['edad_max'], errors='coerce').fillna(200).astype(int)
        d = d[(d['edad_min'] <= int(edad)) & (int(edad) <= d['edad_max'])]
        return None if d.empty else d.iloc[0].to_dict()
    elif 'edad' in d.columns:
        d['edad'] = pd.to_numeric(d['edad'], errors='coerce')
        d = d.dropna(subset=['edad'])
        d['diff'] = (d['edad'].astype(int) - int(edad)).abs()
        return d.sort_values('diff').iloc[0].to_dict()
    return d.iloc[0].to_dict()

def calcular_prima(edad, sexo, fumador, suma, condiciones=[]):
    base_row = buscar_prima(tarifa_base_df, edad, sexo, fumador)
    tasa_por_mil = float(base_row['prima_base']) if base_row and 'prima_base' in base_row else 6.0
    factor_total, desglose = 1.0, [f"Base por edad/sexo/fumador → {tasa_por_mil} ‰"]

    for cond in condiciones:
        enf_row = buscar_prima(tarifa_enf_df, edad, sexo, fumador, enfermedad=cond)
        sal_row = buscar_prima(tarifa_salud_df, edad, sexo, fumador, enfermedad=cond)
        if enf_row and 'factor' in enf_row:
            factor_total *= float(enf_row['factor'])
            desglose.append(f"+ {cond} × {float(enf_row['factor']):.2f} (tabla enfermedades)")
        if sal_row and 'factor' in sal_row:
            factor_total *= float(sal_row['factor'])
            desglose.append(f"+ {cond} × {float(sal_row['factor']):.2f} (tabla salud)")

    prima_pura = (tasa_por_mil/1000) * suma * factor_total
    prima_tarifa = prima_pura * (1+GASTOS_Y_UTILIDAD)
    premio = prima_tarifa * (1+IMP_TASAS_SELLOS)
    return {"tasa_por_mil": tasa_por_mil,"factor_total": factor_total,
            "prima_pura": round(prima_pura,2),"prima_tarifa": round(prima_tarifa,2),
            "premio": round(premio,2),"desglose": desglose}

# ----------------------------
# Factores de riesgo
# ----------------------------
ACTIVIDAD_RIESGO = {'Oficina': 1.00,'Industria': 1.15,'Construccion': 1.30,'Transporte': 1.20}

def factores_riesgo(edad, imc, fumador, actividad):
    up, down = [], []
    if edad >= 60: up.append("Edad ≥ 60")
    elif edad >= 45: up.append("Edad 45–59")
    else: down.append("Edad < 45")
    if imc >= 35: up.append("IMC ≥ 35 (obesidad)")
    elif imc >= 30: up.append("IMC 30–34.9 (sobrepeso)")
    elif imc >= 18.5: down.append("IMC normal (18.5–24.9)")
    else: up.append("IMC < 18.5 (bajo peso)")
    if fumador == 1: up.append("Fumador")
    else: down.append("No fumador")
    if ACTIVIDAD_RIESGO.get(actividad,1.0) > 1.0:
        up.append(f"Actividad de mayor riesgo ({actividad})")
    else:
        down.append(f"Actividad de bajo riesgo ({actividad})")
    return up, down

# ----------------------------
# Interfaz
# ----------------------------
st.title('Agente IA - Seguro de Accidentes Personales')
uploaded = st.file_uploader('Subí un PDF', type=['pdf'])
valores_pdf = extraer_pdf(uploaded) if uploaded else {}

if valores_pdf:
    st.info("📄 Valores extraídos del PDF")
    st.json(valores_pdf)

default_edad = int(valores_pdf.get('edad', 30) or 30)
default_peso = float(valores_pdf.get('peso', 70) or 70)
default_talla = float(valores_pdf.get('talla', 170) or 170)
default_sexo_idx = 0 if valores_pdf.get('sexo', 'M') == 'M' else 1
default_fum = int(valores_pdf.get('fumador', 0) or 0)

st.subheader("Parámetros manuales")
edad = st.number_input("Edad", 18, 100, default_edad)
peso = st.number_input("Peso (kg)", 40.0, 200.0, default_peso)
talla = st.number_input("Talla (cm)", 120.0, 220.0, default_talla)
actividad = st.selectbox("Actividad", ['Oficina','Industria','Construccion','Transporte'])
sexo = st.selectbox("Sexo", ['M','F'], index=default_sexo_idx)
fumador = 1 if st.selectbox("¿Es fumador?", ['No','Sí'], index=default_fum, key="sel_fumador") == 'Sí' else 0
suma_asegurada = st.number_input("Suma asegurada (ARS)", 50000, 1000000, 100000)

# Condiciones detectadas
auto_conditions = []
if valores_pdf.get('diabetes',0) == 1: auto_conditions.append('Diabetes')
if valores_pdf.get('hipertension',0) == 1: auto_conditions.append('Hipertensión')
if valores_pdf.get('enf_renal',0) == 1: auto_conditions.append('Enfermedad renal')
if valores_pdf.get('asma',0) == 1: auto_conditions.append('Asma/Respiratoria')

# FIX corregido: ahora sí usa dropna() sobre Series
condiciones_disp = sorted(set(
    tarifa_enf_df.get('enfermedad', pd.Series([], dtype=str)).dropna().astype(str).tolist() +
    tarifa_salud_df.get('enfermedad', pd.Series([], dtype=str)).dropna().astype(str).tolist() +
    auto_conditions
))
condiciones_sel = st.multiselect("Condiciones de salud", condiciones_disp, default=auto_conditions)

input_df = pd.DataFrame([{'edad':edad,'peso':peso,'talla':talla,
                          'fumador':fumador,'actividad':actividad}])

imc_input = round(peso / ((talla/100)**2), 1)
fact_up, fact_down = factores_riesgo(int(edad), imc_input, int(fumador), actividad)
probas = modelo.predict_proba(input_df)
p_asegurable, p_no_asegurable = probas[0][1], probas[0][0]
decision = "Asegurable" if p_asegurable>=0.5 else "No asegurable"

# Reglas de negocio (NO APTO, edad+fumador, ≥2 enfermedades graves)
alto_riesgo = []
motivos_rechazo = []

# Regla 1: Edad ≥55 y fumador
if edad >= 55 and fumador == 1:
    alto_riesgo.append("Edad ≥55 y Fumador")
    motivos_rechazo.append("Riesgo alto por edad y tabaquismo")

# Regla 2: Informe clínico NO APTO en PDF
if valores_pdf.get('no_apto', 0) == 1:
    alto_riesgo.append("Resultado clínico: NO APTO")
    motivos_rechazo.append("Rechazado por informe clínico NO APTO")

# Regla 3: Dos o más condiciones graves
condiciones_graves = [
    valores_pdf.get('diabetes', 0),
    valores_pdf.get('hipertension', 0),
    valores_pdf.get('enf_renal', 0),
    valores_pdf.get('asma', 0)
]
if sum(condiciones_graves) >= 2:
    alto_riesgo.append("≥2 condiciones graves detectadas")
    motivos_rechazo.append("Rechazado por múltiples enfermedades graves")

# Aplicar fuerza de decisión si alguna regla se cumple
if alto_riesgo:
    decision, p_asegurable, p_no_asegurable = "No asegurable", 0.0, 1.0

    # Mostrar aviso global
    st.warning("Caso de ALTO RIESGO detectado: " + ", ".join(alto_riesgo))

    # Mostrar motivos detallados
    st.markdown("#### Motivos de rechazo")
    for motivo in motivos_rechazo:
        st.error("• " + motivo)

# Prima siempre calculada
prima = calcular_prima(int(edad), sexo, int(fumador), float(suma_asegurada), condiciones_sel)
prima_no_fum = calcular_prima(int(edad), sexo, 0, float(suma_asegurada), condiciones_sel)

impacto_fumador_abs = prima['premio'] - prima_no_fum['premio']
impacto_fumador_pct = (impacto_fumador_abs / prima_no_fum['premio'] * 100) if prima_no_fum['premio'] > 0 else 0

penalizado = False
if decision.startswith("No asegurable"):
    penalizado = True
    prima['prima_pura'] = round(prima['prima_pura'] * 1.5, 2)
    prima['prima_tarifa'] = round(prima['prima_tarifa'] * 1.5, 2)
    prima['premio'] = round(prima['premio'] * 1.5, 2)
    prima['desglose'].append("Penalización por NO APTO / no asegurable ×1.5")

# ----------------------------
# Resultados
# ----------------------------
st.subheader("Resultados")
col1, col2 = st.columns(2)
with col1:
    if decision.startswith("Asegurable"):
        st.success(f"Decisión: {decision}")
    else:
        st.error(f"Decisión: {decision}")
    st.metric("Riesgo (P no asegurable)", f"{p_no_asegurable*100:.2f}%")
    st.progress(int(round(p_no_asegurable*100)))
with col2:
    st.metric("Probabilidad (P asegurable)", f"{p_asegurable*100:.2f}%")
    st.caption(f"IMC calculado: **{imc_input}**")

st.markdown("### Prima estimada (Argentina)")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Prima pura", f"ARS {prima['prima_pura']:,}")
with c2: st.metric("Prima de tarifa", f"ARS {prima['prima_tarifa']:,}")
with c3:
    if penalizado:
        st.error(f"Premio final penalizado: ARS {prima['premio']:,}")
    else:
        st.metric("Premio final", f"ARS {prima['premio']:,}")

with st.expander("Ver desglose de factores de prima"):
    for item in prima['desglose']:
        st.write("• " + item)
    st.caption("Fórmula: prima_pura = tasa‰/1000 × suma × factores  →  prima_tarifa = prima_pura × (1+gastos)  →  premio = prima_tarifa × (1+impuestos).")

if fumador == 1:
    st.warning(f"Impacto por ser **fumador**: +ARS {impacto_fumador_abs:,.0f} (≈ {impacto_fumador_pct:.1f}%)")

st.markdown("### Factores que influyeron en la decisión")
left, right = st.columns(2)
with left:
    st.markdown("**Aumentan el riesgo**")
    if fact_up:
        for f in fact_up: st.write("• " + f)
    else:
        st.write("• —")
with right:
    st.markdown("**Reducen el riesgo**")
    if fact_down:
        for f in fact_down: st.write("• " + f)
    else:
        st.write("• —")

# ----------------------------
# Matriz de Confusión y métricas
# ----------------------------
st.markdown("### Matriz de Confusión del modelo")
cm = np.array(metrics['confusion_matrix'])
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No asegurable", "Asegurable"],
            yticklabels=["No asegurable", "Asegurable"], ax=ax)
ax.set_title("Matriz de Confusión", fontsize=12, pad=10)
ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real")
st.pyplot(fig)

st.markdown("### Métricas del modelo")
st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
st.write(f"Accuracy: {metrics['classification_report']['accuracy']:.3f}")

