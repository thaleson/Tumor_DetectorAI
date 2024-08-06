import json
import streamlit as st
from streamlit_lottie import st_lottie

def load_lottie_animation(file_path):
    """Carregar a animação Lottie a partir do arquivo JSON"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"Erro ao decodificar o arquivo JSON: {file_path}")
        return None

def show_home():
    # Configurar o título da página
    st.title("Bem-vindo ao Sistema de Detecção de Tumor Cerebral! 🧠")

    # Adicionar subtítulo
    st.subheader("Olá! Saiba mais sobre o nosso projeto de detecção de tumores cerebrais 👋")

    # Carregar animações
    animation_1 = load_lottie_animation("animaçoes/animationbrain.json")  # Substitua pelo caminho correto do seu arquivo de animação
    animation_2 = load_lottie_animation("animaçoes/animationbrain2.json")  # Substitua pelo caminho correto do seu arquivo de animação

    if animation_1 and animation_2:
        # Configurar layout em colunas
        col1, col2 = st.columns(2)

        # Conteúdo da coluna 1
        with col1:
            st_lottie(animation_1, height=350, width=350, key="animation1")
            st.markdown(
                """
                <div style='margin-top: 10px;'>
                    <h5 style='text-align: justify;'>Este projeto utiliza técnicas avançadas de aprendizado de máquina para detectar tumores cerebrais em imagens de ressonância magnética. Com uma acurácia de 71.47%, nosso modelo é capaz de identificar diferentes tipos de tumores cerebrais com alta precisão.</h5>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Conteúdo da coluna 2
        with col2:
            st.markdown(
                """
                <div style='margin-top: 10px;'>
                    <h5 style='text-align: justify;'>Nosso sistema pode classificar imagens em quatro categorias: glioma, meningioma, pituitário e sem tumor. Utilizamos a arquitetura MobileNetV2 para um balanceamento ótimo entre desempenho e eficiência.</h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            st_lottie(animation_2, height=350, width=350, key="animation2")

        # Adicionar um aviso
        st.markdown(
            """
            <div style='background-color: #d4edda; padding: 15px; border-radius: 8px;'>
                <h4 style='color: #155724;'>Aviso:</h4>
                <p style='color: #155724;'>Este projeto é uma demonstração das capacidades de detecção de tumores cerebrais. A precisão pode variar e o sistema deve ser usado como uma ferramenta de suporte, não substituindo uma avaliação médica profissional.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Não foi possível carregar as animações.")

if __name__ == "__main__":
    show_home()
