""" API za predviđanje budućih transakcija korisnika """

import math

import numpy as np
import pandas as pd
from flask import request, Flask
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


@app.route("/api", methods=["POST"])
def predict_data():
    """ Ruta za pristup API-ju """

    req = request.get_json()

    # Obrada JSON objekta
    df = pd.DataFrame(
        {"Total": np.array(req["Total"]), "Change": np.array(req["Change"])}
    )

    # Izračun percentilne volatilnosti promjene salda
    df["PCT_change"] = df["Change"] / df["Total"] * 100
    predict_column = "Total"

    # Generiranje stupca sa oznakama
    predict_shift = int(math.ceil(0.08 * len(df)))
    df["label"] = df[predict_column].shift(-predict_shift)

    X = np.array(df.drop(["label"], 1))
    X = preprocessing.scale(X)
    X_recent = X[:predict_shift]
    X = X[:-predict_shift]

    df.dropna(inplace=True)
    y = np.array(df["label"])

    # Odvajanje podataka za treniranje i testiranje
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    prediction_set = clf.predict(X_recent)
    if math.isnan(confidence):
        confidence = -9999

    result = {
        "prediction": list(prediction_set),
        "confidence": float(confidence),
        "prediction_shift": predict_shift,
    }

    return result, 201
