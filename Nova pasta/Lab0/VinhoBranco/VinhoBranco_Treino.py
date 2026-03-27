import pandas as pd
import pickle as p1
from sklearn import linear_model

data = pd.read_csv("./winequality-white.csv",
sep=";")

train_data=data[:1000]
data_X=train_data.iloc[:,0:11]
data_Y=train_data.iloc[:,11:12]
#print(train_data.columns)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('./white-wine_quality_predictor', 'wb')
print("white-wine_quality_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)



