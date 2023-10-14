from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pandas as pd
def modify_column(x):
    if x == "Ac2":
        x = 0
    elif x == "Alloc80":
        x = 1
    elif x == "BackProp":
        x = 2
    elif x == "Bayes":
        x = 3
    elif x == "BayesTree":
        x = 4
    elif x == "C4.5":
        x = 5
    elif x == "CART":
        x = 6
    elif x == "Cal5":
        x = 7
    elif x == "Cascade":
        x = 8
    elif x == "Castle":
        x = 9
    elif x == "Cn2":
        x = 10
    elif x == "Default":
        x = 11
    elif x == "Dipol92":
        x = 12
    elif x == "Discrim":
        x = 13
    elif x == "ITrule":
        x = 14
    elif x == "IndCART":
        x = 15
    elif x == "KNN":
        x = 16
    elif x == "Kohonen":
        x = 17
    elif x == "LVQ":
        x = 18
    elif x == "LogDisc":
        x = 19
    elif x == "NewId":
        x = 20
    elif x == "QuaDisc":
        x = 21
    elif x == "RBF":
        x = 22
    elif x == "Smart":
        x = 23
    return x
data = pd.read_csv('meta.csv', header = None)
data = data.drop(columns = [0, 8, 10, 12])
data[20] = data[20].apply(modify_column)
etichete = data[21]
data_train = data[:384]
etichete_train = etichete[:384]
data_test = data[384:]
etichete_test = etichete[384:]
regr = ensemble.RandomForestRegressor(n_estimators = 10, max_samples = 0.85, max_features = 0.8)
regr.fit(data_train, etichete_train)
predictie = regr.predict(data_test)
MSE = mean_squared_error(etichete_test, predictie)
MSE1 = 0
for i in range (144):
    MSE1 = MSE1 + (etichete_test[i + 384] - predictie[i]) * (etichete_test[i + 384] - predictie[i])
MSE1 = MSE1 / 144;
print(f"MSE: {MSE:.2f}")
print(f"MSE1: {MSE1:.2f}")