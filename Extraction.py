import pandas as pd
import os
 
class DataExtraction:
    def _init_(self, file_path="traffic.csv"):
        self.file_path = file_path
 
    def extract(self):
        print("Extracting traffic data...")
        df = pd.read_csv(self.file_path)
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/raw_traffic.csv", index=False)
        print(f"Extracted {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())
        return df