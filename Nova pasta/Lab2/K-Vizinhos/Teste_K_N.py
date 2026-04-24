import pandas as pd
import pickle as p1
from sklearn import linear_model
import numpy as np

with open("DigitsDadosteste.pkl", "rb") as Dt:
    DigitsDadosteste = p1.load(Dt)

digits_data_X_test = DigitsDadosteste[0]
digits_data_Y_test = DigitsDadosteste[1]

print(type(digits_data_X_test))
digits_model_loaded = p1.load(open('digits_predictor', 'rb'))
print("Coefficients (digits): \n", digits_model_loaded.coef_)

digits_y_pred = digits_model_loaded.predict(digits_data_X_test)

# Arredondar previsões para o inteiro mais próximo para comparar com as classes reais dos dígitos
digits_y_pred_rounded = np.round(digits_y_pred).astype(int)

right_digits = 0
wrong_digits = 0
total_digits = len(digits_data_Y_test)

# Assumindo que digits_data_Y_test é um DataFrame com uma única coluna 'class'
for i in range(total_digits):
    if digits_y_pred_rounded[i][0] == digits_data_Y_test.iloc[i, 0]:
        right_digits += 1
    else:
        wrong_digits += 1

print("accuraccy1 (digits)= ", right_digits / total_digits)
print("accuraccy2 (digits)= ", wrong_digits / total_digits)