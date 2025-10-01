from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

# ------------------------
# Configuración inicial
# ------------------------
styles = getSampleStyleSheet()
styleH = styles["Heading1"]
styleH.spaceAfter = 12
styleH.alignment = 1  # centrado
styleN = styles["Normal"]
styleN.spaceAfter = 12

# ------------------------
# Documento
# ------------------------
doc = SimpleDocTemplate("reporte_seguro.pdf", pagesize=A4,
                        rightMargin=2*cm, leftMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

Story = []

# ------------------------
# Portada
# ------------------------
Story.append(Spacer(1, 4*cm))
Story.append(Paragraph("🤖 Agente IA - Seguro de Accidentes Personales", styleH))
Story.append(Spacer(1, 1*cm))
Story.append(Paragraph("Autores: <b>Solange Areco y Milagros Lencina</b>", styleN))
Story.append(Spacer(1, 10*cm))
Story.append(Paragraph("Trabajo Práctico - Challenge One", styleN))
Story.append(Paragraph("Fecha: Septiembre 2025", styleN))
Story.append(PageBreak())

# ------------------------
# Introducción
# ------------------------
Story.append(Paragraph("1. Introducción", styleH))
Story.append(Paragraph(
    "Este proyecto implementa un agente de inteligencia artificial que evalúa "
    "la asegurabilidad de personas para seguros de accidentes personales. "
    "El sistema combina un modelo de Machine Learning con reglas de negocio "
    "determinísticas y permite calcular una prima estimada según tablas locales "
    "de Argentina.", styleN))

# ------------------------
# Datasets
# ------------------------
Story.append(Paragraph("2. Datasets utilizados", styleH))
Story.append(Paragraph("• ENNyS2 (Encuesta Nacional de Nutrición y Salud, Argentina 2018-2019) - datos.gob.ar - Licencia Pública.", styleN))
Story.append(Paragraph("• Kaggle Insurance Dataset - Kaggle.com - Licencia CC0 (uso libre).", styleN))
Story.append(Paragraph("• Dataset sintético generado con Pandas/Numpy para ampliar diversidad de casos.", styleN))

# ------------------------
# Pipeline técnico
# ------------------------
Story.append(Paragraph("3. Pipeline del modelo", styleH))
Story.append(Paragraph(
    "1. Ingestión: carga robusta de CSV (UTF-8, latin-1, cp1252).\n"
    "2. Limpieza y features: edad, peso, talla, IMC, tabaquismo, enfermedades.\n"
    "3. Entrenamiento: RandomForestClassifier con balance de clases.\n"
    "4. Evaluación: métricas de accuracy, ROC AUC y matriz de confusión.\n"
    "5. Exportación: modelo guardado con joblib para reutilización en Streamlit.", styleN))

# ------------------------
# Reglas de negocio
# ------------------------
Story.append(Paragraph("4. Reglas de negocio aplicadas", styleH))
Story.append(Paragraph(
    "• Edad ≥55 y fumador → No asegurable.\n"
    "• Informe PDF con 'NO APTO' → No asegurable.\n"
    "• Dos o más enfermedades graves (diabetes, hipertensión, renal, respiratoria) → No asegurable.\n"
    "• En caso de rechazo, se aplica penalización ×1.5 en la prima final.", styleN))

# ------------------------
# Métricas del modelo
# ------------------------
Story.append(Paragraph("5. Métricas del modelo", styleH))
Story.append(Paragraph(
    "El modelo RandomForest alcanza un accuracy ≈ 0.98 y un ROC AUC ≈ 0.999 en validación. "
    "La matriz de confusión y métricas detalladas están integradas en la interfaz de Streamlit.", styleN))

# ------------------------
# Análisis ético
# ------------------------
Story.append(Paragraph("6. Análisis ético y sesgos", styleH))
Story.append(Paragraph(
    "El uso de variables como edad, sexo y enfermedades crónicas puede introducir sesgos "
    "en la decisión de asegurabilidad. Se recomienda monitorear fairness y aplicar métricas "
    "adicionales para evitar discriminación no justificada desde el punto de vista actuarial.", styleN))

# ------------------------
# Conclusiones
# ------------------------
Story.append(Paragraph("7. Conclusiones", styleH))
Story.append(Paragraph(
    "El agente IA cumple con los objetivos planteados: predicción binaria de asegurabilidad, "
    "probabilidad de riesgo y cálculo de prima estimada en base a normativa argentina. "
    "Como pasos futuros, se sugiere ampliar las fuentes de datos, mejorar el análisis ético "
    "y desplegar el sistema en la nube para pruebas con usuarios reales.", styleN))

# ------------------------
# Guardar PDF
# ------------------------
doc.build(Story)
print("✅ PDF final generado: reporte_seguro.pdf")

