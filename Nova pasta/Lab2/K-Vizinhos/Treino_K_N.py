import pandas as pd
import pickle as p1
from sklearn import linear_model
import numpy as np


from ucimlrepo import fetch_ucirepo

# Obter o conjunto de dados 'optical_recognition_of_handwritten_digits'
digits_data = fetch_ucirepo(id=80)

print("nº total de dados (digits)", len(digits_data.data.features))
print("nº total de parâmetros (digits)", len(np.array(digits_data.data.features[:1])[0]), " + 1")

M_digits = len(digits_data.data.features)
m_digits = int(2 * M_digits / 3)  # Usar 2/3 para treino, semelhante ao modelo do vinho
print("nº total de dados de treino (digits)", m_digits)

digits_data_X = digits_data.data.features[:m_digits]
digits_data_Y = digits_data.data.targets[:m_digits]

# Dados de teste restantes para os dígitos
digits_datateste_X = digits_data.data.features[m_digits + 1:]
digits_datateste_Y = digits_data.data.targets[m_digits + 1:]
DigitsDadosteste = [digits_datateste_X, digits_datateste_Y]

with open("DigitsDadosteste.pkl", "wb") as Dt:
    p1.dump(DigitsDadosteste, Dt)

regr_digits = linear_model.LinearRegression()
digits_model = regr_digits.fit(digits_data_X, digits_data_Y)

# Salvar o modelo de dígitos
with open('digits_predictor', 'wb') as PickleModeloDigits:
    p1.dump(digits_model, PickleModeloDigits)
print("digits_predictor guardado com sucesso!")