import streamlit as st
import os
from agente_SMN import responder_usuario_para_streamlit

st.set_page_config(page_title="Asistente Meteorológico SMN", page_icon="🌤️", layout="wide")
st.title("🌤️ Asistente Meteorológico - SMN")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Renderear historial de conversación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Mostrar imágenes del historial con tamaño controlado (500px)
        if "imagenes" in message and message["imagenes"]:
            for img_path in message["imagenes"]:
                if os.path.exists(img_path):
                    st.image(img_path, width=500)

        # Mostrar PDF del historial
        if "pdf_path" in message and message["pdf_path"] and os.path.exists(message["pdf_path"]):
            with open(message["pdf_path"], "rb") as f:
                st.download_button(
                    label="📄 Descargar Informe PDF",
                    data=f,
                    file_name=os.path.basename(message["pdf_path"]),
                    mime="application/pdf",
                    key=f"pdf_{message['pdf_path']}"
                )

# 2. Entrada de usuario y nueva consulta
if prompt := st.chat_input("Escribí tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Procesando consulta y ejecutando herramientas..."):
            try:
                res = responder_usuario_para_streamlit(prompt, st.session_state.messages[:-1])
                texto_res = res["texto"]
                pdf_res = res["pdf_path"]
                imagenes_res = res["imagenes"]
            except Exception as e:
                texto_res = f"❌ Ocurrió un error al procesar la respuesta: {e}"
                pdf_res = None
                imagenes_res = []

            # Mostrar respuesta de texto
            st.markdown(texto_res)

            # Mostrar imágenes con tamaño controlado (500px)
            if imagenes_res:
                for img_path in imagenes_res:
                    if os.path.exists(img_path):
                        st.image(img_path, width=500)

            # Mostrar descarga de PDF
            if pdf_res and os.path.exists(pdf_res):
                with open(pdf_res, "rb") as f:
                    st.download_button(
                        label="📄 Descargar Informe PDF",
                        data=f,
                        file_name=os.path.basename(pdf_res),
                        mime="application/pdf"
                    )

    # Guardar en estado de sesión
    st.session_state.messages.append({
        "role": "assistant", 
        "content": texto_res, 
        "pdf_path": pdf_res,
        "imagenes": imagenes_res
    })