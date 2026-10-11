
import os
import streamlit as st

from pathlib import Path
from PIL import Image as PILImage
from agente_SMN import responder_usuario_para_streamlit


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Asistente Meteorológico SMN",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌤️ Asistente Meteorológico - SMN")
st.caption(
    "Consultá datos meteorológicos, analizá temperaturas, "
    "generá gráficos y elaborá informes."
)


# =========================================================
# ESTADO DE SESIÓN
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "consulta_sugerida" not in st.session_state:
    st.session_state.consulta_sugerida = None


# =========================================================
# CONFIGURACIÓN DE IMÁGENES
# =========================================================

ANCHO_IMAGEN_HORIZONTAL = 700
ANCHO_IMAGEN_VERTICAL = 420


# =========================================================
# CONSULTAS DE EJEMPLO
# =========================================================

EJEMPLOS = {
    "Informes": [
        "Generá un informe meteorológico completo para Rosario Aero.",
        "Generá un informe meteorológico para Córdoba Aero."
    ],

    "Análisis de temperaturas": [
        "Graficá las temperaturas de Rosario Aero durante septiembre de 2026.",
        "Compará las temperaturas de Rosario Aero y Córdoba Aero durante septiembre de 2026."
    ],

    "Estaciones meteorológicas": [
        "¿Cuáles son las estaciones meteorológicas más cercanas a Rosario Aero?",
        "Generá un mapa de ubicación de la estación Rosario Aero."
    ],

    "Climatología": [
        "Generá un climograma histórico de Rosario Aero.",
        "¿Qué información climatológica histórica tenés disponible para Rosario Aero?"
    ]
}


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def seleccionar_consulta(consulta):
    """Guarda una consulta sugerida para procesarla."""
    st.session_state.consulta_sugerida = consulta


def mostrar_imagenes(imagenes, indice_mensaje):
    """
    Muestra imágenes centradas, adaptando su tamaño
    a la orientación, con descarga individual.
    """

    for i, img_path in enumerate(imagenes):

        ruta = Path(img_path)

        if not ruta.is_file():
            continue

        # Determinar orientación
        with PILImage.open(ruta) as imagen:
            ancho_px, alto_px = imagen.size

        es_vertical = alto_px > ancho_px

        # Columnas para centrar el contenido
        if es_vertical:
            columnas = st.columns([2, 3, 2])
            ancho_maximo = ANCHO_IMAGEN_VERTICAL
        else:
            columnas = st.columns([1, 5, 1])
            ancho_maximo = ANCHO_IMAGEN_HORIZONTAL

        with columnas[1]:

            st.image(
                str(ruta),
                width=ancho_maximo
            )

            with open(ruta, "rb") as archivo:
                imagen_bytes = archivo.read()

            st.download_button(
                label="Descargar imagen",
                data=imagen_bytes,
                file_name=ruta.name,
                mime=(
                    "image/png"
                    if ruta.suffix.lower() == ".png"
                    else "image/jpeg"
                    if ruta.suffix.lower() in [".jpg", ".jpeg"]
                    else "application/octet-stream"
                ),
                key=f"imagen_{indice_mensaje}_{i}",
                icon=":material/download:",
                on_click="ignore"
            )


def mostrar_pdf(pdf_path, indice_mensaje):
    """Muestra el botón de descarga del informe PDF."""

    if not pdf_path:
        return

    ruta = Path(pdf_path)

    if not ruta.is_file():
        return

    with open(ruta, "rb") as archivo:
        pdf_bytes = archivo.read()

    st.download_button(
        label="Descargar informe PDF",
        data=pdf_bytes,
        file_name=ruta.name,
        mime="application/pdf",
        key=f"pdf_{indice_mensaje}",
        icon=":material/picture_as_pdf:",
        on_click="ignore"
    )


def mostrar_mensaje(mensaje, indice_mensaje):
    """Renderiza un mensaje con texto, imágenes y PDF."""

    with st.chat_message(mensaje["role"]):

        st.markdown(mensaje["content"])

        imagenes = mensaje.get("imagenes", [])

        if imagenes:
            mostrar_imagenes(
                imagenes,
                indice_mensaje
            )

        pdf_path = mensaje.get("pdf_path")

        if pdf_path:
            mostrar_pdf(
                pdf_path,
                indice_mensaje
            )


# =========================================================
# BARRA LATERAL
# =========================================================

with st.sidebar:

    st.header("Guía del asistente")

    st.markdown(
        """
        Este asistente permite consultar y analizar
        información meteorológica de estaciones del
        Servicio Meteorológico Nacional.

        **¿Qué podés hacer?**

        - Consultar datos meteorológicos.
        - Analizar temperaturas históricas.
        - Comparar estaciones.
        - Generar gráficos y mapas.
        - Consultar información documental.
        - Generar informes meteorológicos en PDF.
        """
    )

    st.divider()

    st.subheader("Consultas de ejemplo")

    for categoria, consultas in EJEMPLOS.items():

        with st.expander(categoria):

            for i, consulta in enumerate(consultas):

                st.button(
                    consulta,
                    key=f"sidebar_{categoria}_{i}",
                    on_click=seleccionar_consulta,
                    args=(consulta,),
                    width="stretch"
                )

    st.divider()

    st.caption(
        "Las respuestas dependen de la disponibilidad "
        "de datos y de las herramientas meteorológicas."
    )


# =========================================================
# HISTORIAL DE CONVERSACIÓN
# =========================================================

for indice, mensaje in enumerate(st.session_state.messages):
    mostrar_mensaje(mensaje, indice)


# =========================================================
# PANTALLA INICIAL
# =========================================================

if not st.session_state.messages:

    st.markdown("### ¿Qué te gustaría consultar?")

    st.write(
        "Podés escribir una pregunta o comenzar "
        "con alguno de estos ejemplos."
    )

    ejemplos_inicio = [
        (
            "Informe meteorológico",
            "Generá un informe meteorológico completo para Rosario Aero."
        ),
        (
            "Comparación de temperaturas",
            "Compará las temperaturas de Rosario Aero y Córdoba Aero durante septiembre de 2026."
        ),
        (
            "Climatología histórica",
            "Generá un climograma histórico de Rosario Aero."
        )
    ]

    columnas = st.columns(3)

    for i, (titulo, consulta) in enumerate(ejemplos_inicio):

        with columnas[i]:

            with st.container(border=True):

                st.markdown(f"**{titulo}**")
                st.caption(consulta)

                st.button(
                    "Probar consulta",
                    key=f"inicio_{i}",
                    on_click=seleccionar_consulta,
                    args=(consulta,),
                    width="stretch"
                )


# =========================================================
# ENTRADA DEL USUARIO
# =========================================================

prompt_chat = st.chat_input(
    "Escribí tu consulta meteorológica..."
)

# Priorizar consultas seleccionadas desde los ejemplos
prompt = st.session_state.consulta_sugerida or prompt_chat

st.session_state.consulta_sugerida = None


# =========================================================
# PROCESAMIENTO DE CONSULTAS
# =========================================================

if prompt:

    # Guardar mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar respuesta del agente
    with st.chat_message("assistant"):

        with st.spinner(
            "Procesando consulta y ejecutando herramientas..."
        ):

            try:

                res = responder_usuario_para_streamlit(
                    prompt,
                    st.session_state.messages[:-1]
                )

                texto_res = res.get("texto", "")
                pdf_res = res.get("pdf_path")
                imagenes_res = res.get("imagenes") or []

            except Exception as e:

                texto_res = (
                    "Ocurrió un error al procesar "
                    f"la consulta: {e}"
                )

                pdf_res = None
                imagenes_res = []

        # Mostrar respuesta textual
        st.markdown(texto_res)

        # Mostrar imágenes
        indice_mensaje = len(st.session_state.messages)

        if imagenes_res:
            mostrar_imagenes(
                imagenes_res,
                indice_mensaje
            )

        # Mostrar PDF
        if pdf_res:
            mostrar_pdf(
                pdf_res,
                indice_mensaje
            )

    # Guardar respuesta en historial
    st.session_state.messages.append({
        "role": "assistant",
        "content": texto_res,
        "pdf_path": pdf_res,
        "imagenes": imagenes_res
    })
