import pandas as pd
import os

def split_data_for_time_series(data, test_size=0.2):
    # Ordenar os dados pelo tempo (assumindo que 'ds' é a coluna de data)
    data_sorted = data.sort_values(by='ds')

    # Calcular o índice para divisão entre treino e teste
    split_index = int((1 - test_size) * len(data_sorted))

    # Dividir os dados
    train_data = data_sorted.iloc[:split_index]
    test_data = data_sorted.iloc[split_index:]

    return train_data, test_data


df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'data', 'Processed_data', 'processed_data.csv')))

# Supondo que 'ds' é a coluna de data e 'y' é o alvo
X = df[['ds']]  # Recursos, se houver mais colunas de features
y = df['y']     # Variável alvo

# Dividir os dados para treino e teste
train_data, test_data = split_data_for_time_series(df)

# Exibir tamanhos dos conjuntos de treino e teste
print("\n")
print(f"Quantidade de dados de treino: {len(train_data)}")
print(f"Quantidade de dados de teste: {len(test_data)}")
print("\n")

# Salvar dados de treino e teste em arquivos CSV
train_data.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_data_train_test', 'train_data.csv')))
test_data.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_data_train_test', 'test_data.csv')))

print("Dados de treino e teste salvos com sucesso.")
print("\n")