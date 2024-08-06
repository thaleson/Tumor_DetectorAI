import streamlit as st

def show_about():
    st.title("Sobre o Projeto")

    st.markdown(
        """
        ## Detecção de Tumor Cerebral com Aprendizado de Máquina 🧠

        Este projeto tem como objetivo desenvolver um sistema de detecção de tumores cerebrais utilizando técnicas avançadas de aprendizado de máquina. A seguir, detalhamos todo o processo de desenvolvimento e treinamento do modelo.

        ### Arquitetura do Modelo

        Utilizamos a arquitetura MobileNetV2 para o desenvolvimento do nosso modelo. A MobileNetV2 é uma rede neural convolucional eficiente, projetada para dispositivos móveis e aplicações com restrições de recursos. Ela é conhecida por seu equilíbrio entre desempenho e eficiência, sendo capaz de realizar classificações precisas com menos parâmetros em comparação com outras arquiteturas mais pesadas.

        ### Processo de Treinamento

        1. **Coleta de Dados**: 
           - Utilizamos um conjunto de dados contendo imagens de ressonância magnética (MRI) de quatro categorias: glioma, meningioma, pituitário e sem tumor.
           - Cada categoria contém 999 imagens, totalizando 3.996 imagens para treinamento e validação.

        2. **Pré-processamento dos Dados**:
           - As imagens foram redimensionadas para 128x128 pixels para garantir uniformidade e compatibilidade com a entrada da MobileNetV2.
           - Aplicamos técnicas de data augmentation para aumentar a diversidade do conjunto de treinamento e melhorar a robustez do modelo. As técnicas incluíram rotações, zoom, deslocamentos e flips horizontais.

        3. **Divisão dos Dados**:
           - Os dados foram divididos em conjuntos de treinamento (80%) e validação (20%) utilizando a classe `ImageDataGenerator` do TensorFlow.

        4. **Transfer Learning**:
           - Inicializamos o modelo MobileNetV2 pré-treinado com pesos da ImageNet e removemos as camadas de saída originais.
           - Adicionamos camadas densas personalizadas para ajustar o modelo à tarefa específica de detecção de tumor cerebral.

        5. **Treinamento**:
           - O modelo foi treinado utilizando o otimizador Adam com uma taxa de aprendizado reduzida para promover uma convergência mais suave.
           - Implementamos o callback de Early Stopping para interromper o treinamento quando a perda de validação não melhorasse após 5 épocas consecutivas.

        ### Desempenho do Modelo

        Após o treinamento, nosso modelo alcançou uma acurácia de 71.47% no conjunto de teste. Esta precisão indica que o modelo é capaz de identificar tumores cerebrais em imagens de MRI com uma alta taxa de acerto, embora haja espaço para melhorias adicionais.

        ### Desafios e Aprendizados

        Durante o desenvolvimento deste projeto, enfrentamos diversos desafios, como o balanceamento das classes e a necessidade de um grande volume de dados para melhorar a precisão do modelo. Através da utilização de técnicas de data augmentation e transfer learning, conseguimos mitigar alguns desses desafios e alcançar resultados satisfatórios.

        ### Conclusão

        Este projeto demonstra o potencial do aprendizado de máquina na área da saúde, especificamente na detecção de tumores cerebrais. Continuaremos trabalhando para aprimorar o modelo e explorar novas técnicas para melhorar ainda mais a precisão e a robustez do sistema.

        Agradecemos pelo seu interesse no nosso projeto e esperamos que ele possa contribuir para avanços significativos na área da saúde.

        ### Código de Treinamento

        Abaixo está o código utilizado para o treinamento do modelo:
        """
    )

    st.code("""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Carregar e pré-processar os dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Construir o modelo
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas base do modelo
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinar o modelo
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Salvar o modelo treinado
model.save('brain_tumor_detection_model.h5')
    """, language="python")

if __name__ == "__main__":
    show_about()
