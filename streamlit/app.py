import streamlit as st
import requests
import base64
from PIL import Image
import io

# ------------------ Config ------------------ #
st.set_page_config(page_title="XSight", layout="centered")

# ------------------ UI ------------------ #
st.title("XSight: Radiography Analyzer")
st.markdown("Upload a chest X-ray and provide patient info to analyze potential pathologies.")

uploaded_file = st.file_uploader("Upload Radiograph", type=['png', 'jpg', 'jpeg'])

col1, col2 = st.columns(2)
with col1:
    patient_age = st.number_input("👤 Patient Age", min_value=0, max_value=120, value=45)
    pixel_spacing_x = st.slider("📏 Image Width Detail (mm)", 0.00, 0.20, 0.10, step=0.01)
with col2:
    patient_sex = st.selectbox("⚧️ Sex", ['M', 'F'])
    pixel_spacing_y = st.slider("📏 Image Height Detail (mm)", 0.00, 0.20, 0.10, step=0.01)

view_position = st.selectbox("📐 View Position", ['Front View', 'Back View'])

# ------------------ API Call ------------------ #
if uploaded_file and st.button("🧪 Analyze"):
    files = {'file': uploaded_file}
    data = {
        'patient_age': patient_age,
        'patient_sex': patient_sex,
        'view_position': view_position,
        'pixel_spacing_x': pixel_spacing_x,
        'pixel_spacing_y': pixel_spacing_y
    }

    with st.spinner("🔎 Analyzing in progress..."):
        try:
            response = requests.post('http://localhost:8000/process_all', files=files, data=data)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")
            st.stop()

    # ------------------ Display Results ------------------ #
    st.success("✅ Analysis complete!")

    st.subheader("Detected Pathologies")
    st.write(f"**→** {result['pathologies']}")

    if result.get("heatmap"):
        st.subheader("🔥 Heatmap (Grad-CAM)")
        heatmap_bytes = base64.b64decode(result["heatmap"])
        heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
        st.image(heatmap_image, use_column_width=True)

    if result.get("confidence"):
        st.subheader("Confidence")
        st.write(f"Confidence score: **{result['confidence']:.2f}**")
