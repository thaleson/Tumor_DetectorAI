# 🧠 Projeto de Detecção de Tumor Cerebral

Este projeto é um sistema de detecção de tumores cerebrais desenvolvido com técnicas de aprendizado de máquina e visualização interativa utilizando Streamlit.

## 📚 Sobre o Projeto

O objetivo deste projeto é criar um modelo que possa classificar imagens de ressonância magnética (MRI) de cérebros em quatro categorias:
- Glioma
- Meningioma
- Pituitário
- Sem tumor

### 🔍 Arquitetura do Modelo

Utilizamos a arquitetura MobileNetV2 para o nosso modelo. Esta rede neural convolucional é eficiente e ideal para dispositivos móveis e aplicações com restrições de recursos.

### 📊 Desempenho

Após o treinamento, nosso modelo alcançou uma acurácia de **71.47%** no conjunto de teste.

## 🛠️ Tecnologias Utilizadas

- Python
- TensorFlow
- Streamlit
- MobileNetV2

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- TensorFlow 2.13.0
- Virtualenv ou Conda (recomendado)

### Passo a Passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/thaleson/Tumor_DetectorAI
   cd IAmedico
   ```

2. **Crie e ative um ambiente virtual**
   - Usando `virtualenv`:
     ```bash
     python -m venv venv
     source venv/bin/activate  # No Windows use: .\venv\Scripts\activate
     ```
   - Usando `conda`:
     ```bash
     conda create --name ia_tumor python=3.8
     conda activate ia_tumor
     ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação**
   ```bash
   streamlit run main.py
   ```

## 🌟 Funcionalidades

### Home 🏠

A página inicial oferece uma introdução ao projeto e explica a motivação por trás dele.

### Previsão 🔍

A página de previsão permite que os usuários façam upload de imagens de MRI e obtenham previsões sobre a presença e o tipo de tumor cerebral.

### Sobre o Projeto ℹ️

Uma descrição detalhada do projeto, incluindo o processo de treinamento, arquitetura do modelo e desempenho.

## 🎨 Estilo

Para um visual atraente e profissional, utilizamos um arquivo `styles.css` personalizado.

## ⚠️ Aviso

Este modelo não é 100% preciso e não deve ser usado como substituto para diagnósticos médicos. Consulte sempre um profissional de saúde para uma avaliação adequada.


## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

Esperamos que este projeto possa contribuir para avanços significativos na área da saúde e detecção de tumores cerebrais. Obrigado por seu interesse!


