import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io
import time

# Carregar o modelo salvo
model_path = 'models/brain_tumor_detection_model_compressed.h5'
model = tf.keras.models.load_model(model_path)

# Função para carregar e preparar a imagem
def prepare_image(img):
    img = image.load_img(img, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Função para realizar a previsão
def predict_image(img):
    prepared_image = prepare_image(img)
    predictions = model.predict(prepared_image)
    return predictions

# Função para verificar se a imagem é uma ressonância magnética do cérebro
def is_brain_mri_image(img):
    try:
        # Verificar formato da imagem
        if img.format not in ('JPEG', 'PNG'):
            return False

        # Verificar dimensões da imagem (exemplo: entre 100x100 e 1000x1000 pixels)
        width, height = img.size
        if not (100 <= width <= 1000 and 100 <= height <= 1000):
            return False

        # Verificar se a imagem é em escala de cinza ou RGB
        if img.mode not in ('L', 'RGB'):
            return False

        # Simulação de uma verificação adicional para a imagem do cérebro
        # No mundo real, seria necessário um modelo de classificação mais avançado
        # para identificar se a imagem é de uma ressonância magnética do cérebro especificamente.
        # Aqui, usamos apenas as verificações básicas.
        return True
    except Exception as e:
        return False

# Classes
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def show_prediction():
    st.title("Previsão de Tumor Cerebral 🧠")

    # Carregar imagem
    uploaded_file = st.file_uploader("Escolha uma imagem de ressonância magnética ou raio X", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if not is_brain_mri_image(img):
            st.warning("Isso não parece ser uma ressonância magnética do cérebro ou uma imagem apropriada. Por favor, carregue uma imagem relevante.")
            return

        st.image(uploaded_file, caption='Imagem carregada', use_column_width=True)
        
        if st.button("Fazer Previsão"):
            with st.spinner('Realizando previsão...'):
                time.sleep(2)
                predictions = predict_image(uploaded_file)
                predicted_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_index]
                predicted_probability = predictions[0][predicted_index] * 100

                # Mensagens de resultados
                if predicted_class == 'notumor':
                    st.success(f'Parabéns! Isso não é um tumor. Probabilidade: {predicted_probability:.2f}%')
                else:
                    st.error(f'Isso é possivelmente um {predicted_class} com uma probabilidade de {predicted_probability:.2f}%')

                st.markdown(
                    """
                    <div style='background-color: #f8d7da; padding: 15px; border-radius: 8px;'>
                        <h4 style='color: #721c24;'>Aviso:</h4>
                        <p style='color: #721c24;'>Este modelo não é 100% preciso. Procure um médico para uma avaliação mais detalhada.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    show_prediction()
