import pandas as pd
import pickle as p1
from sklearn import linear_model

data = pd.read_csv("C:/Users/anilt/Documents/MATÉRIAS_2_SEMESTRE/Aprendizagem Automática/winequality-white.csv",
sep=";")

right=0
wrong=0
total=0
for x in z_pred["quality"]:
    z=int(x)
    total=total+1
    if z==0:
        right=right+1
    else:
        wrong=wrong+1
    print("accuraccy1= ",right/total,"accuraccy2= ",wrong/total)