import joblib
import pandas as pd
import numpy as np


class GenderPredictor:

    def __init__(self):

        self.model = joblib.load("models/gender_model.pkl")

        self.scaler = joblib.load("models/scaler.pkl")

        self.encoder = joblib.load("models/label_encoder.pkl")

    def predict(self, csv_file):

        df = pd.read_csv(csv_file)

        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        X = self.scaler.transform(df)

        prediction = self.model.predict(X)

        probability = self.model.predict_proba(X)

        gender = self.encoder.inverse_transform(prediction)

        confidence = np.max(probability)

        return gender[0], round(confidence * 100, 2)
