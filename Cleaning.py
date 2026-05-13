import pandas as pd
import os
 
class DataCleaning:
    def _init_(self, df):
        self.df = df
 
    def clean(self):
        print("Cleaning data...")
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        self.df.columns = self.df.columns.str.strip().str.lower()
        os.makedirs("data/cleaned", exist_ok=True)
        self.df.to_csv("data/cleaned/cleaned_data.csv", index=False)
        print(f"Cleaned data shape: {self.df.shape}")
        return self.df
