import pandas as pd
import pickle as p1
from sklearn import linear_model
import numpy as np

# Carregar o dataset localmente
data = pd.read_csv("digits.csv", sep=";")

# Verificar a estrutura do dataset
print("Colunas disponíveis:", data.columns.tolist())
print("Shape do dataset:", data.shape)

# Assumindo que a última coluna é o target
X = data.iloc[:, :-1].values  # features
y = data.iloc[:, -1].values   # target

# Número total de dados
M = len(data)
print("Nº total de dados:", M)

# Número de parâmetros (features)
n_parametros = X.shape[1]
print("Nº total de parâmetros:", n_parametros, "+ 1")

# Definir tamanho do treino (2/3 dos dados)
m = int(2 * M / 3)
print("Nº total de dados de treino:", m)

# Separar dados de treino
train_X = X[:m]
train_y = y[:m]

# Dados de teste
teste_X = X[m:]
teste_y = y[m:]

# Salvar dados de teste
Dadosteste = [teste_X, teste_y]
with open("Dadosteste.pkl", "wb") as Dt:
    p1.dump(Dadosteste, Dt)

# Criar e treinar o modelo
regr = linear_model.LinearRegression()
Modelo = regr.fit(train_X, train_y)

# Guardar o modelo
with open('digits_predictor', 'wb') as PickleModelo:
    p1.dump(Modelo, PickleModelo)

print("Modelo guardado com sucesso!")
print("Dimensão do modelo: ", Modelo.coef_.shape)

print("="*70)

