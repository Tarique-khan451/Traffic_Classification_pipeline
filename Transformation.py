import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle, os
 
class DataTransformation:
    def _init_(self, df, target_column):
        self.df = df
        self.target_column = target_column
 
    def transform(self):
        print("Transforming data...")
        le = LabelEncoder()
        scaler = StandardScaler()
        for col in self.df.select_dtypes(include="object").columns:
            if col != self.target_column:
                self.df[col] = le.fit_transform(self.df[col])
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_scaled = scaler.fit_transform(X)
        os.makedirs("data/transformed", exist_ok=True)
        pickle.dump(scaler, open("data/transformed/scaler.pkl", "wb"))
        print("Transformation complete.")
        return X_scaled, y
