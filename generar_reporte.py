from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from datetime import datetime

# Configuración del documento
doc = SimpleDocTemplate("reporte_seguro.pdf", pagesize=A4)
styles = getSampleStyleSheet()

# Personalización de estilos
styles.add(ParagraphStyle(name='Titulo', fontSize=20, alignment=1, spaceAfter=20))
styles.add(ParagraphStyle(name='Subtitulo', fontSize=14, alignment=1, spaceAfter=10))
styles.add(ParagraphStyle(name='Pie', fontSize=8, alignment=2, textColor="grey"))

story = []

# ----------------------------
# Portada
# ----------------------------
story.append(Paragraph("Agente IA - Seguro de Accidentes Personales", styles['Titulo']))
story.append(Paragraph("Informe Técnico - Evaluación y Resultados", styles['Subtitulo']))
story.append(Spacer(1, 50))
story.append(Paragraph(f"Autor: Sol", styles['Normal']))
story.append(Paragraph(f"Fecha: {datetime.today().strftime('%d/%m/%Y')}", styles['Normal']))
story.append(PageBreak())

# ----------------------------
# Secciones descriptivas
# ----------------------------
secciones = [
    ("1. Introducción", 
     "Este informe documenta el desarrollo del Agente de IA para evaluar asegurabilidad en seguros de accidentes personales."),
    ("2. Fuentes de datos",
     "Se integraron datasets: ENNyS2, Insurance (Kaggle), y datos sintéticos generados con Pandas/Numpy."),
    ("3. Pipeline del modelo",
     "Incluye preprocesamiento, Random Forest, métricas de validación y reglas de negocio determinísticas."),
    ("4. Factores de riesgo",
     "Edad, IMC, tabaquismo y tipo de actividad influyen en el riesgo."),
    ("5. Evaluación del modelo",
     "Se presentan métricas del modelo: matriz de confusión, ROC AUC, precisión y recall."),
]

# Agregar capturas
for i, (titulo, texto) in enumerate(secciones, start=1):
    story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading2']))
    story.append(Paragraph(texto, styles['Normal']))
    story.append(Spacer(1, 12))

    try:
        img_path = f"captura{i}.png"
        story.append(Image(img_path, width=12*cm, height=8*cm))
        story.append(Spacer(1, 24))
    except Exception as e:
        story.append(Paragraph(f"Imagen {i} no encontrada: {e}", styles['Normal']))

# ----------------------------
# Pie de página
# ----------------------------
def pie_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawString(2*cm, 1*cm, "Proyecto: Agente IA - Seguro de Accidentes Personales")
    canvas.drawRightString(19*cm, 1*cm, f"Página {doc.page}")
    canvas.restoreState()

# Generar PDF
doc.build(story, onLaterPages=pie_pagina, onFirstPage=pie_pagina)
print("✅ PDF con portada y formato generado: reporte_seguro.pdf")
