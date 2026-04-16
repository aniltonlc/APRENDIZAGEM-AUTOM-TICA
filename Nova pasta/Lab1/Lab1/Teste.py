# Teste

import pandas as pd
import pickle as p1
from sklearn import linear_model
import numpy as np

# Carregar dados de teste
with open("Dadosteste.pkl", "rb") as Dt:
    Dadosteste = p1.load(Dt)

data_X = Dadosteste[0]
data_Y = Dadosteste[1]

print("Tipo de data_X:", type(data_X))
print("Shape de data_X:", data_X.shape)
print("Shape de data_Y:", data_Y.shape)

# Carregar modelo
Modelogravado = p1.load(open('digits_predictor', 'rb'))
print("Coeficientes:\n", Modelogravado.coef_)

# Fazer predições
y_pred = Modelogravado.predict(data_X)

# Calcular erro (diferença entre predição e valor real)
z_pred = y_pred - data_Y

# Para classificação, precisamos arredondar as predições para o dígito mais próximo
# e limitar entre 0 e 9 (dígitos válidos)
y_pred_rounded = np.round(y_pred).astype(int)
y_pred_rounded = np.clip(y_pred_rounded, 0, 9)  # garantir que está entre 0 e 9

# Calcular acurácia
right = 0
wrong = 0
total = len(data_Y)

for i in range(total):
    if y_pred_rounded[i] == data_Y[i]:
        right += 1
    else:
        wrong += 1

print("Eficácia de acertos:", (right/total)*100)
print("Taxa de erro:", (wrong/total)*100)

# Alternativa mais eficiente usando NumPy
accuracy = np.mean(y_pred_rounded == data_Y)
print("\nAcurácia (método NumPy):", accuracy)

# Métricas adicionais
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(data_Y, y_pred)
r2 = r2_score(data_Y, y_pred)

print("\nMSE (Erro Quadrático Médio):", mse)
print("R² Score:", r2)

# Mostrar primeiras 10 predições vs valores reais
"""print("\nPrimeiras 10 predições vs reais:")
for i in range(min(10, total)):
    print(f"Predição: {y_pred_rounded[i]:.0f}, Real: {data_Y[i]:.0f}, Valor bruto: {y_pred[i]:.2f}")"""