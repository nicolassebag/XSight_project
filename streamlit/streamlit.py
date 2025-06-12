import streamlit as st
import requests
from PIL import Image
import base64
import streamlit.components.v1 as components

API_URL = "https://xsightdocker-1047363581301.europe-west1.run.app/analyze"  # Update if deployed

# Set app to wide mode
st.set_page_config(page_title="XSight", layout="wide")

# # Background image using HTML injection
# components.html(
#     """
#     <style>
#     body {
#         background-image: url('https://images.unsplash.com/photo-1588776814546-ec7e4b6dff53?auto=format&fit=crop&w=1650&q=80');
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#     }
#     </style>
#     """,
#     height=0,
# )


# Background image using CSS injection
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://imageio.forbes.com/specials-images/imageserve/66f9624039565d35b6ecb4b3/Radiology-Doctor-working-diagnose-treatment-virtual-Human-Lungs-and-long-Covid-19-on/960x0.jpg?format=jpg&width=960");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App Title
st.title("🧠 XSight")

st.markdown(
    """
Bienvenue sur **XSight** – votre assistant intelligent pour l'analyse des radiographies ! 🩻
Téléversez une image pour obtenir une **analyse automatique**, incluant des visualisations avancées et une estimation des pathologies détectées.

---

👈 **Commencez en téléversant une radio** au format JPG, JPEG ou PNG ci-dessous.
"""
)

uploaded_file = st.file_uploader("📤 **Charger une radio**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Radio chargée", width=400)

    with st.spinner("🔍 Analyse de l'image..."):
        try:
            response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})

            if response.status_code == 200:
                html = response.text

                # Extract base64 images from HTML
                heatmap_base64 = html.split('src="data:image/jpeg;base64,')[1].split('"')[0]
                prob_plot_base64 = html.split('src="data:image/png;base64,')[1].split('"')[0]

                col1, col2 = st.columns(2)

                with col2:
                    st.subheader("🔥 Visualisation: Prédiction principale")
                    st.image(f"data:image/jpeg;base64,{heatmap_base64}", use_container_width=True)

                with col1:
                    st.subheader("📊 Potentielles pathologies")
                    st.image(f"data:image/png;base64,{prob_plot_base64}", use_container_width=True)

                # 👉 Comparison section
                st.markdown("### 🆚 Comparaison avec annotation médicale")
                doctor_file = st.file_uploader("📥 Charger l'annotation radiologue", type=["jpg", "jpeg", "png"])

                if doctor_file is not None:
                    doctor_image = Image.open(doctor_file).convert("RGB")

                    col_model, col_doctor = st.columns(2)

                    with col_model:
                        st.subheader("🔍 Prédiction XSight")
                        st.image(f"data:image/jpeg;base64,{heatmap_base64}", use_container_width=True)

                    with col_doctor:
                        st.subheader("🩺 Annotation radiologue")
                        st.image(doctor_image, use_container_width=True)
                else:
                    st.info("Veuillez charger une annotation médicale pour la comparaison.")

            else:
                st.error(f"❌ Erreur API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"⚠️ Échec de la connexion à l'API : {str(e)}")
