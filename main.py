import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json

# Configuração da página principal
st.set_page_config(page_title="IA Tumor Detection", page_icon="🧠")

st.markdown(
    f"""
    <style>
    {open("static/styles.css").read()}
    </style>
    """,
    unsafe_allow_html=True
)

# Carregar animações
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

# Carregar animação
animation_path = "animaçoes/animationmain.json"
animation = load_lottie_animation(animation_path)

# Menu de navegação
with st.sidebar:
    # Exibir animação
    if animation:
        st_lottie(animation, height=100, width=270)

    selected = option_menu(
        "Menu", 
        ["Home", "Previsão", "Sobre o Projeto",],
        icons=['house', 'bar-chart', 'info-circle'],
        menu_icon="cast", 
        default_index=0
    )

    # Badges
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between;">
            <div>
                <a href="https://github.com/thaleson" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" width="100" />
                </a>
            </div>
            <div>
                <a href="https://www.linkedin.com/in/thaleson-silva-9298a0296/" target="_blank">
                    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="100" />
                </a>
            </div>
            <div>
                <a href="mailto:thaleson177@gmail.com" target="_blank">
                    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" width="80" />
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Navegação para as páginas
if selected == "Home":
    from pages.nav import home
    home.show_home()
elif selected == "Previsão":
    from pages.nav import predict
    predict.show_prediction()
elif selected == "Sobre o Projeto":
    from pages.nav import about
    about.show_about()

