import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def marks_prediction(hrs):
    X = pd.read_excel("Linear_X_Train.xlsx")
    y = pd.read_excel("Linear_Y_Train.xlsx")

    X = X.values
    y = y.values

    model = LinearRegression()
    model.fit(X,y)

    X_test = np.array(hrs, dtype=float)
    X_test = X_test.reshape((1,-1))

    return model.predict(X_test)[0][0]
