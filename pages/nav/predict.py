import pandas as pd

# Criando um DataFrame a partir de um dicionário
data = {
    'Nome': ['Ana', 'Bruno', 'Carlos'],
    'Idade': [23, 35, 45],
    'Cidade': ['São Paulo', 'Rio de Janeiro', 'Curitiba']
}

df = pd.DataFrame(data)

# Exibindo o DataFrame
print(df)

# Leitura de um arquivo CSV
df = pd.read_csv('arquivo.csv')

# Filtrando dados
df_filtrado = df[df['Idade'] > 30]

# Calculando a média de uma coluna
media_idade = df['Idade'].mean()

print(f'Média de Idade: {media_idade}')
